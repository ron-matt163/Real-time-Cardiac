#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

import os, numpy as np
import re, time
import matplotlib.pyplot as plt
import torch
from biosppy.signals import ecg

from sklearn.metrics import classification_report
from collections import Counter
from sklearn.utils import resample
import pywt

from scipy.interpolate import interp1d
from scipy.signal import find_peaks, butter, filtfilt

import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data import Dataset



class ECGDataset(Dataset):
    def __init__(self, features, labels):
        """
        features: A list or array of cycle feature dictionaries (including 'id' and 'features').
        labels: A numpy array or list of labels corresponding to each cycle feature dictionary.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_dict = self.features[idx]
        # Assuming each feature in the dictionary is already a numpy array of the same length
        # or has been appropriately padded/processed before this point.

        features = torch.tensor()
        for i in feature_dict:

            # Keeping intervals separate, consider each as a channel in a 1D CNN input
            feature_tensor = torch.stack([
                torch.tensor(i['features']['ecg_segment'], dtype=torch.float32),
                torch.tensor(i['features']['rr_interval'], dtype=torch.float32),
                torch.tensor(i['features']['pq_interval'], dtype=torch.float32),
                torch.tensor(i['features']['qt_interval'], dtype=torch.float32),
                torch.tensor(i['features']['pr_interval'], dtype=torch.float32),
                torch.tensor(i['features']['st_interval'], dtype=torch.float32)
            ], dim=0)  # This stacks the tensors along a new dimension, treat each interval as a separate channel

            features = torch.cat((features, feature_tensor))

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label


class PytorchExperimentLogger(object):
    def __init__(self, saveDir, fileName, ShowTerminal=False):
        self.saveFile = saveDir + r"/" + fileName + ".txt"
        self.ShowTerminal = ShowTerminal

    def print(self, strT):
        # Convert strT to a string if it's not
        strT = str(strT)

        # Print to terminal if ShowTerminal is True
        if self.ShowTerminal:
            print(strT)

        # Append the string to the file
        with open(self.saveFile, 'a') as f:
            f.writelines(strT + '\n')



exp_logger = PytorchExperimentLogger('./log', "log", ShowTerminal=True)



def extract_features_per_cycle(ecg_signal, r_peaks, q_peaks, s_peaks, p_peaks, t_peaks, sampling_rate, id):
    """
    Extract detailed ECG features for each cycle, including raw segments, amplitudes,
    intervals, HRV metrics (SDNN, RMSSD), heart rate, and kSQI.
    """
    print(f"ECG signal for id: {id}, {ecg_signal}")
    cnt = 0
    cycle_data_list = []
    for i in range(len(p_peaks)):

        # print(i, " / ", len(p_peaks))

        if i >= len(q_peaks) or i >= len(r_peaks) or i >= len(s_peaks) or i >= len(t_peaks):
            cnt += 1
            continue

        if i == len(p_peaks) - 1:
            ecg_segment = resample_template(ecg_signal[p_peaks[i]: -1], 50)
            rr_interval = resample_template(ecg_signal[r_peaks[i]: -1], 50)
        else:
            ecg_segment = resample_template(ecg_signal[p_peaks[i]: p_peaks[i+1]], 50)
            rr_interval = resample_template(ecg_signal[r_peaks[i]: r_peaks[i+1]], 50)

        pq_interval = resample_template(ecg_signal[p_peaks[i]: q_peaks[i]], 50)
        qt_interval = resample_template(ecg_signal[q_peaks[i]: t_peaks[i]], 50)
        pr_interval = resample_template(ecg_signal[p_peaks[i]: r_peaks[i]], 50)
        st_interval = resample_template(ecg_signal[s_peaks[i]: t_peaks[i]], 50)

        # p_amp = np.full(50, ecg_signal[p_peaks[i]])
        # q_amp = np.full(50, ecg_signal[q_peaks[i]])
        # r_amp = np.full(50, ecg_signal[r_peaks[i]])
        # s_amp = np.full(50, ecg_signal[s_peaks[i]])
        # t_amp = np.full(50, ecg_signal[t_peaks[i]])
        if len(ecg_segment) != 50 or len(rr_interval) != 50 or len(pq_interval) != 50 or len(qt_interval) != 50 or len(pr_interval) != 50 or len(st_interval) != 50:
            print(f"LENGTH INCONSISTENCY! for id = {id}: ecgs {len(ecg_segment)}, rr_interval {len(rr_interval)}, pq {len(pq_interval)}, qt {len(qt_interval)}, pr {len(pr_interval)} st {len(st_interval)}")
        # print(np.array([ecg_segment, rr_interval, pq_interval, qt_interval, pr_interval, st_interval]).shape)

        # Bundle each cycle's features and id into a dictionary
        cycle_features = {
            # 'id': id,
            'features': {
                'ecg_segment': ecg_segment,
                'rr_interval': rr_interval,
                'pq_interval': pq_interval,
                'qt_interval': qt_interval,
                'pr_interval': pr_interval,
                'st_interval': st_interval
                # 'p_amp': p_amp,
                # 'q_amp': q_amp,
                # 'r_amp': r_amp,
                # 's_amp': s_amp,
                # 't_amp': t_amp
            }
        }

        cycle_data_list.append(cycle_features)

    return cycle_data_list, cnt



def process_ecg(recording, frequency, id):

    lead_one = dwt_denoise(recording[0])
    r_peaks, q_peaks, s_peaks, p_peaks, t_peaks = hamilton_detector_with_qs_pt(lead_one, frequency)
    print(f"P peaks: {p_peaks}, Q peaks: {q_peaks}, R peaks: {r_peaks}, S peaks: {s_peaks}, T peaks: {t_peaks}")

    # plt.figure(figsize=(10, 6))
    # plt.plot(lead_one, label=f'ECG Lead One', color='black')

    # # Plot the peaks using different markers and colors
    # plt.plot(r_peaks, lead_one[r_peaks], 'ro', label='R Peaks')
    # plt.plot(q_peaks, lead_one[q_peaks], 'bo', label='Q Peaks')
    # plt.plot(s_peaks, lead_one[s_peaks], 'go', label='S Peaks')
    # plt.plot(p_peaks, lead_one[p_peaks], 'mo', label='P Peaks')
    # plt.plot(t_peaks, lead_one[t_peaks], 'yo', label='T Peaks')

    # plt.legend()
    # plt.title(f'ECG Signal with Detected Peaks; recording ID: {id}')
    # plt.xlabel('Time steps')
    # plt.ylabel('Amplitude')
    # plt.show()

    features, cnt = extract_features_per_cycle(lead_one, r_peaks, q_peaks, s_peaks, p_peaks, t_peaks, frequency, id)

    return features, cnt


# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

def strip_extension(filename):
    return filename.split('.')[0]

# (Re)sort leads using the standard order of leads for the standard twelve-lead ECG.
def sort_leads(leads):
    x = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    leads = sorted(leads, key=lambda lead: (x.index(lead) if lead in x else len(x) + leads.index(lead)))
    return tuple(leads)

# Find header and recording files.
def find_challenge_files(data_directory):
    header_files = list()
    recording_files = list()
    for f in sorted(os.listdir(data_directory)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.hea':
            header_file = os.path.join(data_directory, root + '.hea')
            recording_file = os.path.join(data_directory, root + '.mat')
            if os.path.isfile(header_file) and os.path.isfile(recording_file):
                header_files.append(header_file)
                recording_files.append(recording_file)
    return header_files, recording_files

# Load header file as a string.
def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording

# Choose leads from the recording file.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording

# Get recording ID.
def get_recording_id(header):
    recording_id = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                recording_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return recording_id

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get age from header.
def get_age(header):
    age = None
    for l in header.split('\n'):
        if l.startswith('#Age') or l.startswith('# Age'):
            try:
                age = float(l.split(': ')[1].strip())
            except:
                age = float('nan')
    return age

# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('#Sex') or l.startswith('# Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get number of leads from header.
def get_num_leads(header):
    num_leads = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_leads = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_leads

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get number of samples from header.
def get_num_samples(header):
    num_samples = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_samples = float(l.split(' ')[3])
            except:
                pass
        else:
            break
    return num_samples

# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    gain_str = entries[2]
                    gain = float(re.search(r'(\d+(\.\d+)?)(?=\(0\)/mV)', gain_str).group(0))
                    adc_gains[j] = gain
                except:
                    pass
        else:
            break
    return adc_gains

# Get baselines from header.
def get_baselines(header, leads):
    baselines = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    baselines[j] = float(entries[4])
                except:
                    pass
        else:
            break
    return baselines

# Get labels from header.
def get_labels(header):
    labels = list()
    for l in header.split('\n'):
        if l.startswith('#Dx') or l.startswith('# Dx'):

            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels

# Save outputs from model.
def save_outputs(output_file, recording_id, classes, labels, probabilities):
    # Format the model outputs.
    recording_string = '#{}'.format(recording_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = recording_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Save the model outputs.
    with open(output_file, 'w') as f:
        f.write(output_string)

# Load outputs from model.
def load_outputs(output_file):
    with open(output_file, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                recording_id = l[1:] if len(l)>1 else None
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(entry.strip() for entry in l.split(','))
            elif i==3:
                probabilities = tuple(float(entry) if is_finite_number(entry) else float('nan') for entry in l.split(','))
            else:
                break
    return recording_id, classes, labels, probabilities


def plot_ecg_recording(recording_processed, leads, sample_rate):
    num_leads, num_samples = recording_processed.shape
    time = np.arange(num_samples) / sample_rate

    for i in range(num_leads):

        plt.figure(figsize=(12, 10))
        # plt.subplot(num_leads, 1, i + 1)
        plt.plot(time, recording_processed[i])
        plt.title(leads[i])
        plt.ylabel('Amplitude (mV)')
        plt.xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

def find_subfolders(directory):
    subfolders = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subfolders

def find_all_challenge_files(main_directory):
    all_header_files = []
    all_recording_files = []

    # print(main_directory)
    subfolders = find_subfolders(main_directory)
    for subfolder in subfolders:
        header_files, recording_files = find_challenge_files(subfolder)
        all_header_files.extend(header_files)
        all_recording_files.extend(recording_files)

    return all_header_files, all_recording_files


def one_hot_to_labels(one_hot_vectors):
    """
    Convert one-hot encoded vectors to class labels.

    Args:
    one_hot_vectors (np.array): A 2D numpy array of one-hot encoded vectors.

    Returns:
    np.array: A 1D numpy array of class labels.
    """
    return np.argmax(one_hot_vectors, axis=1)

def report_classification(all_labels, all_preds, num_classes):
    """
    Generate a classification report.

    Args:
    all_labels (np.array): 2D array of one-hot encoded true labels or 1D array of class indices.
    all_preds (np.array): 1D array of class indices.
    num_classes (int): Number of classes.

    Returns:
    str: Text summary of the precision, recall, F1 score for each class.
    """
    # if all_labels.ndim > 1:
    #     # Convert one-hot encoded labels to class indices
    #     labels = np.argmax(all_labels, axis=1)
    # elif all_labels.ndim == 1:
    #     # Labels are already class indices
    #     labels = all_labels
    # else:
    #     raise ValueError("Labels must be a 1D or 2D array")
    # print(all_preds)
    print("label: ", all_labels[0], all_preds[0])
    # Generate classification report
    target_names = [f'Class {i}' for i in range(num_classes)]

    return classification_report(all_labels, all_preds, target_names=target_names)


# Define a function to find outliers
def find_outliers_mean(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return np.mean(outliers) if outliers else 0




def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y



def find_qs_points(ecg_signal, r_peaks, window_size=20):
    q_peaks = []
    s_peaks = []

    for r_peak in r_peaks:
        # Define search windows for Q and S points
        q_window_start = max(0, r_peak - window_size)
        q_window_end = r_peak
        s_window_start = r_peak
        s_window_end = min(len(ecg_signal), r_peak + window_size)

        # Find the minimum value (most negative) within each window as Q and S
        q_peak = np.argmin(ecg_signal[q_window_start:q_window_end]) + q_window_start
        s_peak = np.argmin(ecg_signal[s_window_start:s_window_end]) + s_window_start

        q_peaks.append(q_peak)
        s_peaks.append(s_peak)

    return np.array(q_peaks), np.array(s_peaks)

# def find_p_t_points(ecg_signal, q_peaks, s_peaks, sampling_rate):
#     p_peaks = []
#     t_peaks = []

#     # Adjust these intervals based on the typical ECG characteristics and your specific requirements
#     p_search_interval = int(0.2 * sampling_rate)  # Search for P wave within this interval before the Q peak
#     t_search_interval = int(0.6 * sampling_rate)  # Search for T wave within this interval after the S peak

#     for q_peak, s_peak in zip(q_peaks, s_peaks):
#         # Define search windows for P and T points
#         p_window_start = max(0, q_peak - p_search_interval)
#         p_window_end = q_peak
#         t_window_start = s_peak
#         t_window_end = min(len(ecg_signal), s_peak + t_search_interval)

#         # Assuming P and T waves can be identified by a local maximum (for T wave) and minimum (for P wave) in their respective windows
#         if p_window_end - p_window_start > 0:  # Ensure the window is valid
#             p_peak = np.argmin(ecg_signal[p_window_start:p_window_end]) + p_window_start  # P wave as minimum
#             p_peaks.append(p_peak)

#         if t_window_end - t_window_start > 0:  # Ensure the window is valid
#             t_peak = np.argmax(ecg_signal[t_window_start:t_window_end]) + t_window_start  # T wave as maximum
#             t_peaks.append(t_peak)

#     return np.array(p_peaks), np.array(t_peaks)

def find_p_t_points(ecg_signal, q_peaks, s_peaks, r_peaks, sampling_rate):
    p_peaks = []
    t_peaks = []

    # Estimate average R-R interval to adjust search ranges dynamically
    rr_intervals = np.diff(r_peaks)
    avg_rr_interval = np.mean(rr_intervals) if len(rr_intervals) > 0 else sampling_rate  # Fallback to 1 sec if only one R peak

    for i, (q_peak, s_peak) in enumerate(zip(q_peaks, s_peaks)):
        # Adjust P wave search range based on physiological expectations
        # Calculate backward from Q peak to ensure it's within the expected atrial depolarization period
        p_search_start = max(0, q_peak - int(0.2 * sampling_rate))
        p_search_end = q_peak

        # Adjust T wave search range based on average R-R interval and physiological expectations
        # Start from S peak and ensure the search does not exceed the halfway point to the next R peak
        t_search_start = s_peak
        if i < len(r_peaks) - 1:
            t_search_end = s_peak + min(int(0.6 * sampling_rate), (r_peaks[i + 1] - s_peak) // 2)
        else:
            t_search_end = s_peak + int(0.6 * sampling_rate)

        # Detect P peak as the maximum within its range
        if p_search_end > p_search_start:
            p_segment = ecg_signal[p_search_start:p_search_end]
            if len(p_segment) > 0:
                p_peak = np.argmax(p_segment) + p_search_start
                p_peaks.append(p_peak)

        # Detect T peak as the maximum within its range
        if t_search_end > t_search_start:
            t_segment = ecg_signal[t_search_start:t_search_end]
            if len(t_segment) > 0:
                t_peak = np.argmax(t_segment) + t_search_start
                t_peaks.append(t_peak)

    return np.array(p_peaks), np.array(t_peaks)





def hamilton_detector_with_qs(ecg_signal, sampling_rate):
    # Assuming hamilton_detector function is defined as before and returns r_peaks
    r_peaks_seg = ecg.hamilton_segmenter(ecg_signal, sampling_rate)  # Use the previously defined R peak detection
    r_peaks = r_peaks_seg['rpeaks']
    # print(r_peaks)

    # Now, find Q and S points around each R peak
    q_peaks, s_peaks = find_qs_points(ecg_signal, r_peaks)

    return r_peaks, q_peaks, s_peaks

def hamilton_detector_with_qs_pt(ecg_signal, sampling_rate):
    r_peaks, q_peaks, s_peaks = hamilton_detector_with_qs(ecg_signal, sampling_rate)  # Existing detection

    # Find P and T points around each R peak
    p_peaks, t_peaks = find_p_t_points(ecg_signal, q_peaks, s_peaks, r_peaks, sampling_rate)

    return r_peaks, q_peaks, s_peaks, p_peaks, t_peaks



def is_valid_template(template):
    return template is not None and isinstance(template, np.ndarray) and template.size > 0


def one_hot_encode(labels, all_labels):
    label_vector = np.zeros(len(all_labels), dtype=int)
    label_to_index = {label: index for index, label in enumerate(all_labels)}

    for label in labels:
        if label in label_to_index:
            label_vector[label_to_index[label]] = 1
    return label_vector



def dwt_denoise(ecg_signal):
    coeffs = pywt.wavedec(ecg_signal, 'db4', level=5)
    threshold = np.sqrt(2*np.log(len(ecg_signal)))
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, 'db4')


def udwt_denoise(ecg_signal):
    coeffs = pywt.swt(ecg_signal, 'db4', level=5)
    threshold = np.sqrt(2*np.log(len(ecg_signal)))
    coeffs = [(pywt.threshold(approx, value=threshold, mode='soft'),
               pywt.threshold(detail, value=threshold, mode='soft')) for approx, detail in coeffs]
    return pywt.iswt(coeffs, 'db4')


def biorthogonal_denoise(ecg_signal):
    coeffs = pywt.wavedec(ecg_signal, 'bior3.3', level=5)
    threshold = np.sqrt(2*np.log(len(ecg_signal)))
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, 'bior3.3')


def ti_wavelet_denoise(ecg_signal):
    coeffs = pywt.wavedec(ecg_signal, 'db4', level=5, mode='per')
    threshold = np.sqrt(2*np.log(len(ecg_signal)))
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, 'db4', mode='per')




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



def resample_template(template, target_length=300):
    # Original length of the template
    original_length = len(template)

    if original_length < 2:
        return np.zeros(target_length)

    # Create a function for linear interpolation
    # 'x' is the original indices, 'y' is the template data
    interpolation_function = interp1d(np.linspace(0, 1, original_length), template, kind='linear')

    # New indices for the target length
    new_indices = np.linspace(0, 1, target_length)

    # Use the interpolation function to get data at new indices
    resampled_template = interpolation_function(new_indices)


    # if resampled_template.ndim == 1:
    #     return resampled_template.tolist()

    return resampled_template


def one_hot_encode_2d_list(data, threshold=0.5):
    one_hot_encoded = []
    for row in data:
        encoded_row = [1 if element > threshold else 0 for element in row]
        one_hot_encoded.append(encoded_row)
    return one_hot_encoded



# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)
    num_correct_recordings = 0
    for i in range(num_recordings):

        label = labels[i, :].astype(float)
        output = outputs[i, :]
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)



# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i])-1 for i in range(num_rows))
    if len(row_lengths)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_finite_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Load weights.
def load_weights(weight_file):
    # Load the table with the weight matrix.
    rows, cols, values = load_table(weight_file)

    # Split the equivalent classes.
    rows = [set(row.split('|')) for row in rows]
    cols = [set(col.split('|')) for col in cols]
    assert(rows == cols)

    # Identify the classes and the weight matrix.
    classes = rows
    weights = values

    return classes, weights

# Compute a modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm=set(['426783006'])):
    num_recordings, num_classes = np.shape(labels)
    print(num_recordings, num_classes)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)

    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool_)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score
