import numpy as np
import os, time
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helper_code import *
from sklearn.metrics import accuracy_score, f1_score, classification_report
# from cuml import SVC  # For SVM
from joblib import dump, load

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.model_selection import KFold, GridSearchCV
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from models import *
import xgboost as xgb

from config import config

device = "cuda" if torch.cuda.is_available() else "cpu"


exp_logger = PytorchExperimentLogger('./log', "log", ShowTerminal=True)

scored_classes = ['164889003','164890007','6374002',
                  '426627000','733534002','713427006',
                  '270492004','713426002','39732003',
                  '445118002','164909002','251146004',
                  '698252002','426783006','284470004',
                  '10370003','365413008','427172004',
                  '164947007','111975006','164917005',
                  '47665007','59118001','427393009',
                  '426177001','427084000','63593006',
                  '164934002','59931005','17338001']

map_class = [['733534002', '164909002'],
             ['713427006', '59118001'],
             ['284470004', '63593006'],
             ['427172004', '17338001']]

map1_class = ['733534002', '713427006', '284470004', '427172004']
map2_class = ['164909002', '59118001', '63593006', '17338001']

# Restructure the dataset such that a single ECG cycle forms an input of the model rather that an entire ECG recording
# We don't consider rows that are assigned multiple classes if sinus rhythm is one of those multiple classes.
def split_records_per_cycle(aggregated_features, record_labels):
    # clear_NSR_features contains the features of those ECG cycles that are clearly sinus rhythm
    # unclear_NSR_features contains the features of those ECG cycles that may or may not be sinus rhythm
    clear_NSR_features, unclear_NSR_features = [], []

    if len(aggregated_features) != len(record_labels):
        print("Mismatch in the number of records and labels")
        exit(1)

    uncertain_SR_count = 0
    new_features = []
    new_aggregated_labels = []
    time_spent = []
    for i, record_features in enumerate(aggregated_features):
        start_time = time.time()
        # labels is the set of labels for a single record 
        labels = record_labels[i]
        if '426783006' not in labels:
            for cycle_features in record_features:
                new_features.append(np.array(cycle_features))
                new_aggregated_labels.append(1.0)            
        else:
            if len(labels) == 1:
                for cycle_features in record_features:
                    new_features.append(np.array(cycle_features))
                    new_aggregated_labels.append(0.0)
                    clear_NSR_features.append(np.array(cycle_features))         

            else:
                for cycle_features in record_features:
                    unclear_NSR_features.append(np.array(cycle_features))
                uncertain_SR_count += 1
        time_spent.append(time.time() - start_time)

    print("Uncertain SR count: ", uncertain_SR_count)
    new_features = np.array(new_features)        
    print(f"SHAPES: new_features: {np.array(new_features).shape}, new_aggregated_labels: {np.array(new_aggregated_labels).shape}, clear_NSR_features: {np.array(clear_NSR_features).shape}, unclear_NSR_features: {np.array(unclear_NSR_features).shape}")
    return np.array(new_features), np.array(new_aggregated_labels), np.array(clear_NSR_features), np.array(unclear_NSR_features), time_spent
        



def load_dataset(data_directory):
    print("Start Loading Dataset")
    header_files, recording_files = [], []

    for name in os.listdir(data_directory):
        subfolder = os.path.join(data_directory, name)
        if os.path.isdir(subfolder):
            headers, recordings = find_all_challenge_files(subfolder)
            header_files.extend(headers)
            recording_files.extend(recordings)

    if not len(header_files):
        raise Exception('No data was provided.')

    print("Size of the dataset: ", len(header_files))
    # Accumulate all unique labels from header files
    classes = set()
    for idx in range(len(header_files)):
        header_file = header_files[idx]
        header = load_header(header_file)
        # Get labels and one-hot encode them
        labels = get_labels(header)
        classes.update(labels)

    classes = list(classes)
    classes = [item for item in classes if item not in map2_class]

    scored_indices = [classes.index(item) for item in scored_classes if item in classes]

    error_cnt = 0
    aggregated_features, aggregated_labels, time_spent = [], [], []

    if len(header_files) != len(recording_files):
        print("Header and Recording don't match")

    undersized_list_count = 0
    sinus_rhythm_count = 0
    sole_SR = 0
    not_sole_SR = 0
    record_labels = []
    cyc_in_recording_counts = []

    for idx in range(len(header_files)):
        print(f'{idx}/{len(header_files)}')

        start_time = time.time()
        # Load header and recording
        header_file = header_files[idx]
        recording_file = recording_files[idx]
        header = load_header(header_file)
        recording = load_recording(recording_file, header, ('I'))

        # Extract information from header
        id = get_recording_id(header)
        labels = get_labels(header)
        labels = [map1_class[map2_class.index(item)] if item in map2_class else item for item in labels]
        if '426783006' in labels:
            if len(labels) == 1:
                sole_SR += 1
            else:
                not_sole_SR += 1

        # remove this later
        label_vector = one_hot_encode(labels, classes)
        frequency = get_frequency(header)

        # # Process ECG and extract features
        features, cnt , undersized_list, cyc_in_recording_count = process_ecg(recording, frequency, id)
        cyc_in_recording_counts.append(cyc_in_recording_count)

        if undersized_list:
            undersized_list_count += 1

        end_time = time.time()
        time_interval = end_time-start_time
        time_spent.append(time_interval)
        sole_SR = 0
        not_sole_SR = 0

        error_cnt += cnt

        if not undersized_list:
            aggregated_features.append(features)
            record_labels.append(labels)
            if '426783006' in labels:
                aggregated_labels.append(0.0)
                sinus_rhythm_count += 1
                if len(labels) == 1:
                    sole_SR += 1
                else:
                    not_sole_SR += 1
            else:
                aggregated_labels.append(1.0)
        # COMMENT ENDS HERE

        # for i in range(len(features)):
        #     aggregated_features.append(features[i])
        #     aggregated_labels.append(label_vector)
    print(f"Sole SR: {sole_SR}, Not sole SR: {not_sole_SR}")
    plot_ecg_cycles_histogram(cyc_in_recording_counts)

    # aggregated_features = np.array(aggregated_features)
    # aggregated_labels = np.array(aggregated_labels, dtype=float).reshape(-1,1)

    # print("aggregated_features: ", aggregated_features.shape)
    # print("aggregated_labels: ", aggregated_labels.shape)
    zero_count, one_count = 0, 0
    for label in aggregated_labels:
        if label == 0.0:
            zero_count += 1
        if label == 1.0:
            one_count += 1

    print(f"Zero count = {zero_count}, One count = {one_count}")
    print("error_cnt: ", error_cnt)


    return aggregated_features, aggregated_labels, record_labels, time_spent, error_cnt, scored_indices


def train_model(model, train_loader, num_epochs, save_path):

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):  # num_epochs should be defined
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.reshape(-1, config.feature_size)
            # print("Training inputs: ", str(inputs))
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print("Input shape: ", inputs.shape)
            outputs = model(inputs)
            #print("NEW LOSS FN")
            # change the loss function
            # loss = custom_weighted_loss(outputs, labels, scored_indices)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}; Loss: {loss}')
    torch.save(model.state_dict(), save_path)            

def train_autoencoder_model(model, train_loader, num_epochs, save_path):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss(reduction='sum').to(device)
    loss = None
    # Training loop
    for epoch in range(num_epochs):  # num_epochs should be defined
        model.train()
        train_losses = []
        for i, (inputs, _) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.reshape(-1, config.feature_size)
            # print("AUTOCODER input dimensions in train fn", inputs.shape)
            inputs = inputs.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)
            # print("AUTOCODER output dimensions in train fn", outputs.shape)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f'Epoch: {epoch}; Loss: {mean(train_losses)}')
        print(f"Loss 99.9999th: {np.percentile(train_losses, 99.9999)}, 99.999th: {np.percentile(train_losses, 99.999)}, 99.99th: {np.percentile(train_losses, 99.99)}, 99th: {np.percentile(train_losses, 99)}, 95th: {np.percentile(train_losses, 95)}, 90th: {np.percentile(train_losses, 90)}, 85th: {np.percentile(train_losses, 85)}, 75th: {np.percentile(train_losses, 75)}, 50th: {np.percentile(train_losses, 50)}")
        plot_autoencoder_losses(train_losses, epoch, f"autoenconder_trainloss{epoch}.png")

    torch.save(model.state_dict(), save_path)
    # Find the loss on the training set once again
    test_autoencoder_model(model, train_loader, -1)


def test_model(model, test_loader, scored_indices):
    # Prediction and Evaluation
    start_time = time.time()
    model.eval()
    y_pred = []
    y_test = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.reshape(-1, config.feature_size)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            if outputs.is_cuda:
                outputs = outputs.cpu()

            if labels.is_cuda:
                labels = labels.cpu()

            y_test += list(labels.reshape(-1))

            # # Convert to NumPy array
            # outputs = outputs.numpy()
            # labels = labels.numpy()

            #print("\n\nOutputs in test: ", outputs)
            print("Labels in test: ", labels)

            for _, output in enumerate(outputs):
                if output >= 0.5:
                    y_pred.append(1.0)
                else:
                    y_pred.append(0.0)
            # y_pred.extend(outputs.tolist())
            # y_test.extend(labels.tolist())

    end_time = time.time()

    # y_pred = list(np.array(y_pred).reshape(-1))
    print("y_pred in test: ", y_pred)
    # y_test = list(np.array(labels).reshape(-1))

    # Calculate average time spent per batch
    time_spent_per_batch = (end_time - start_time) / len(test_loader)

    # Calculate average time spent per data point
    time_interval = time_spent_per_batch / 32  # Assuming batch size is 32
    exp_logger.print(f"Average time spent on prediction per data point is {time_interval}s")
    print(f"Size of pred: {len(y_pred)}, size of test: {len(y_test)}")

    print(classification_report(y_test, y_pred))
    # evaluate_model(y_test=y_test, y_pred=y_pred, y_pred_prob=None, scored_indices=scored_indices)

def test_autoencoder_model(model, test_loader, fold=0):
    model.eval()
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.reshape(-1, config.feature_size)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            if outputs.is_cuda:
                outputs = outputs.cpu()

            if labels.is_cuda:
                labels = labels.cpu()

            loss = criterion(outputs, labels)
            predictions.append(outputs.numpy().flatten())
            losses.append(loss.item())

    print(f"Loss 99.9999th: {np.percentile(losses, 99.9999)}, 99.999th: {np.percentile(losses, 99.999)}, 99.99th: {np.percentile(losses, 99.99)}, 99th: {np.percentile(losses, 99)}, 95th: {np.percentile(losses, 95)}, 90th: {np.percentile(losses, 90)}, 85th: {np.percentile(losses, 85)}, 75th: {np.percentile(losses, 75)}, 50th: {np.percentile(losses, 50)}")
    plot_autoencoder_losses(losses, fold)


def train_student_model_with_distillation(teacher_model, student_model, train_loader, device, num_epochs=1, temperature=2.0, alpha=0.5, weights=None, scored_indices=None):
    teacher_model.eval()  # Ensure the teacher model is in evaluation mode
    student_model.train()

    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass of the student
            student_outputs = student_model(inputs)
            # True label loss
            student_loss = custom_weighted_loss(student_outputs, labels, weights, scored_indices)

            # Forward pass of the teacher for soft labels
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # Soften both outputs
            soft_student_outputs = F.softmax(student_outputs / temperature, dim=1)
            soft_teacher_outputs = F.softmax(teacher_outputs / temperature, dim=1)

            # For simplicity, let's say we adapt the custom_weighted_loss to handle soft targets
            # This is a conceptual step; your actual implementation may vary
            distillation_loss = custom_weighted_loss(soft_student_outputs, soft_teacher_outputs, weights, scored_indices)

            # Combined loss
            loss = alpha * student_loss + (1 - alpha) * distillation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss}')


def evaluate_model(y_test, y_pred=None, y_pred_prob=None, scored_indices=None):

    # all_classes = np.load('npy/classes.npy')
    # scored_indices = np.load('npy/scored_indices.npy')
    # map_indices = np.load('npy/map_indices.npy')
    # y_pred = np.load('npy/y_pred.npy')

    y_pred = one_hot_encode_2d_list(y_pred, threshold=0.5)
    y_pred = np.array(y_pred)

    # exp_logger.print(f'before y_test: {y_test.shape}')
    # exp_logger.print(f'before y_pred: {y_pred.shape}')

    y_pred = y_pred[:, scored_indices]
    y_test = y_test[:, scored_indices]

    # exp_logger.print(f'y_pred: {y_pred.shape}')
    # exp_logger.print(f'y_test: {y_test.shape}')

    # Identify the weights and the SNOMED CT code for the sinus rhythm class.
    weights_file = 'csv/weights.csv'
    sinus_rhythm = set(['426783006'])

    # Load the scored classes and the weights for the Challenge metric.
    exp_logger.print('Loading weights...')
    classes, weights = load_weights(weights_file)

    # Load the label and output files.
    exp_logger.print('Loading label and output files...')

    labels = y_test
    binary_outputs = y_pred

    # Evaluate the model by comparing the labels and outputs.
    exp_logger.print('Evaluating model...')

    accuracy = compute_accuracy(labels, binary_outputs)
    exp_logger.print(f'- Accuracy: {accuracy}')

    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)
    exp_logger.print(f'- F-measure: {f_measure}')

    exp_logger.print(f'- Classification Report: {classification_report(y_test, y_pred)}')

    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, sinus_rhythm)
    exp_logger.print(f'- Challenge metric...{challenge_metric}')
    print('Done.')



# Loss Function
def custom_weighted_loss(output, target, weight_matrix, scored_indices):
    """
    Custom loss function with a weight matrix for inter-class relationships, focusing on specific important classes.

    :param output: Predicted probabilities from the model (batch_size x num_classes).
    :param target: True labels (batch_size x num_classes).
    :param weight_matrix: Weight matrix for the important classes (num_important_classes x num_important_classes).
    :param scored_indices: List of indices for important classes.
    :return: Weighted loss value.
    """
    batch_size, num_classes = output.shape
    loss = 0.0

    # Iterate only over important classes and calculate the weighted loss
    for i, class_index in enumerate(scored_indices):
        class_output = output[:, class_index]
        class_target = target[:, class_index]
        class_weight = weight_matrix[i]

        # Calculate binary cross-entropy for this important class
        class_loss = F.binary_cross_entropy(class_output, class_target, reduction='none')

        # Weight this loss by the other important classes
        for j, other_class_index in enumerate(scored_indices):
            other_class_target = target[:, other_class_index]
            loss += class_loss * class_weight[j] * other_class_target

    # return loss.mean() / len(scored_indices)
    return loss.mean()


def main(data_directory,
         model_directory,
         num_epochs = 1,
         is_test=False):

    weights_file = 'csv/weights.csv'

    # classes, weights = load_weights(weights_file)
    # weights = torch.tensor(weights, dtype=torch.float32, requires_grad=False).to(device)

    exp_logger.print(f"=========================")
    # exp_logger.print(f"Process Denoised Technique: {noise} and R-Peak Detection: {peak}")

    aggregated_features, aggregated_labels, record_labels, time_spent, error_cnt, scored_indices = load_dataset(data_directory)
    aggregated_features, aggregated_labels, clear_NSR_features, unclear_NSR_features, restruct_time = split_records_per_cycle(aggregated_features, record_labels)



    # Autoencoder cross validation: MOVE INSIDE A FN TO GENERALIZE IT FOR DIFFERENT MODELS
    # for train_index, test_index in kf.split(clear_NSR_features):

    # exp_logger.print(f"Starting Fold {fold}")
    # X_train, X_test, y_train, y_test = train_test_split(aggregated_features, aggregated_labels, test_size=0.5, random_state=1234+idx)
    X_train, X_test = clear_NSR_features, unclear_NSR_features
    # y_train, y_test = aggregated_labels[train_index], aggregated_labels[test_index]

    # Create custom datasets
    train_dataset = ECGDataset(X_train)
    test_dataset = ECGDataset(X_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    autoencoder_model = RecurrentAutoencoder(config.feature_size, config.autoencoder_embedding_size).to(device)
    # student_model = CNNLSTM1DWithAttentionStudent(num_classes=129).to(device)

    # exp_logger.print("Training the teacher model...")
    # print("Training the autoencoder model")
    model_path = f'{model_directory}/autoencoder_model.pth'
    # train_autoencoder_model(autoencoder_model, train_loader, 500, model_path)

    # # Load the saved model state into the teacher_model instance
    autoencoder_model.load_state_dict(torch.load(model_path, map_location=device))

    exp_logger.print("Testing the autoencoder model...")
    # testing the autoencoder on the test set again
    test_autoencoder_model(autoencoder_model, train_loader)
    # Save the teacher model's state
    torch.save(autoencoder_model.state_dict(), model_path)

    # fold += 1

    print("EXITING AFTER TRAINING AUTOENCODER")
    exit(0)

    # Oversampling the minority class using SMOTE to make up for the class imbalance
    smote = SMOTE(random_state=42)
    aggregated_features, aggregated_labels = smote.fit_resample(aggregated_features, aggregated_labels)
    # smote_fit_resample appears to change the input from 2d to 1d, changing it back here
    aggregated_labels = aggregated_labels.reshape(-1, 1)
    # Shuffle the dataset
    dataset = np.column_stack((aggregated_features, aggregated_labels))
    np.random.shuffle(dataset)

    aggregated_features, aggregated_labels = dataset[:, :-1],  dataset[:, -1]
    aggregated_labels = aggregated_labels.reshape(-1, 1)

    np.save('npy/aggregated_features.npy', aggregated_features)
    np.save('npy/aggregated_labels.npy', aggregated_labels)
    np.save('npy/scored_indices.npy', scored_indices)
    exp_logger.print(f"Number of Errors for R-Peak Extraction: {error_cnt}")
    exp_logger.print(f"Time spent on processing each data is {np.mean(time_spent)}s")

    aggregated_features = np.load('npy/aggregated_features.npy', allow_pickle=True)
    aggregated_labels = np.load('npy/aggregated_labels.npy', allow_pickle=True)
    scored_indices = np.load('npy/scored_indices.npy', allow_pickle=True)

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    iteration = 5
    fold = 1
    one_count = 0
    zero_count = 0

    # Model training
    for train_index, test_index in kf.split(aggregated_features):

        exp_logger.print(f"Starting Fold {fold}")
        # X_train, X_test, y_train, y_test = train_test_split(aggregated_features, aggregated_labels, test_size=0.5, random_state=1234+idx)
        X_train, X_test = aggregated_features[train_index], aggregated_features[test_index]
        y_train, y_test = aggregated_labels[train_index], aggregated_labels[test_index]

        for item in y_train:
            if float(item) == 1.0:
                one_count += 1
            else:
                zero_count += 1

        # # Convert to PyTorch tensors
        # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        # y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        # y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Create custom datasets
        train_dataset = ECGDataset(X_train, y_train)
        test_dataset = ECGDataset(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize the model
        teacher_model = CNNLSTM1DWithAttentionTeacher().to(device)
        # student_model = CNNLSTM1DWithAttentionStudent(num_classes=129).to(device)

        # exp_logger.print("Training the teacher model...")
        print("Training the teacher model")
        model_path = f'{model_directory}/time_ecg.pth'
        train_model(teacher_model, train_loader, num_epochs, model_path)

        # # Load the saved model state into the teacher_model instance
        teacher_model.load_state_dict(torch.load(model_path, map_location=device))

        exp_logger.print("Testing the teacher model...")
        test_model(teacher_model, test_loader, scored_indices)
        # Save the teacher model's state
        torch.save(teacher_model.state_dict(), model_path)



        # Prepare for knowledge distillation
        # exp_logger.print("Training the student model with knowledge distillation...")
        # train_student_model_with_distillation(teacher_model,
        #                                         student_model,
        #                                         train_loader,
        #                                         device,
        #                                         num_epochs=num_epochs,
        #                                         temperature=2.0,
        #                                         alpha=0.5,
        #                                         weights=weights,
        #                                         scored_indices=scored_indices)

        # exp_logger.print("Testing the student model...")
        # model_path = f'{model_directory}/student_model.pth'

        # # Load the saved model state into the teacher_model instance
        # student_model.load_state_dict(torch.load(model_path, map_location=device))
        # test_model(student_model, test_loader, scored_indices)
        # # Optionally save the student model's state
        # torch.save(student_model.state_dict(), f'{model_directory}/student_model.pth')


        fold += 1
    
    print(f"One count: {one_count}, Zero count: {zero_count}")




# Example usage
data_directory = '../physionet.org/files/challenge-2021/1.0.3/training'
model_directory = '..'

main(data_directory,
     model_directory,
     num_epochs=40,
     is_test=False)
