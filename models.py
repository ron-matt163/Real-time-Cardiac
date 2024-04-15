import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F

from attentions import *

class CNNLSTM1DWithAttentionTeacher(nn.Module):
    def __init__(self, lstm_hidden_size=512, lstm_num_layers=3):
        super(CNNLSTM1DWithAttentionTeacher, self).__init__()
        # Initial fully connected layer, output size increased to 64
        self.fc_initial = nn.Linear(5, 64)

        # Conv1d layer after the initial FC, adapting to the increased size
        self.conv_initial = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)

        # Additional fully connected layer after Conv1d transformation, output adjusted for LSTM input
        self.fc_after_conv = nn.Linear(4096, 128)  # Output size adjusted

        # LSTM layer for the sequential part, hidden size increased to 512
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True,
                            bidirectional=True)


        # Convolutional and attention layers for the sequence
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.attention = SelfAttention1D(in_channels=64)

        # Fully connected layers, adjusting input size to accommodate changes
        # self.fc1 = nn.Linear(lstm_hidden_size*2 + 128, 512)  # Adjust the input size for combined output
        self.fc1 = nn.Linear(lstm_hidden_size*2, 512) # Removed '128' since we are not combining the input with x_initial anymore
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # print("x: ", x)
        # Split the input
        # x_initial = x[:, :5].unsqueeze(1)  # Adding channel dimension for Conv1d
        # x_sequence = x[:, 5:]

        # # Process the initial data points
        # x_initial = F.relu(self.fc_initial(x_initial))
        # x_initial = F.relu(self.conv_initial(x_initial))
        # x_initial = torch.flatten(x_initial, start_dim=1)  # Flatten to feed into FC
        # x_initial = F.relu(self.fc_after_conv(x_initial))

        # Process the sequence with Convolutional, Pooling, and Attention
        x_sequence = x.unsqueeze(1)  # Adding channel dimension
        x_sequence = self.pool(F.relu(self.conv1(x_sequence)))
        # print("x_sequence after conv1: ", x_sequence)
        x_sequence = self.pool(F.relu(self.conv2(x_sequence)))
        # print("x_sequence after conv2: ", x_sequence)
        x_sequence = self.pool(F.relu(self.conv3(x_sequence)))
        # print("x_sequence after conv3: ", x_sequence)
        x_sequence = self.attention(x_sequence)
        # print("x_sequence after attention: ", x_sequence)
        # print(x_sequence.shape)
        # Reshape for LSTM
        x_sequence = x_sequence.permute(0, 2, 1)  # [batch_size, seq_length, features]

        # LSTM processing
        x_sequence, _ = self.lstm(x_sequence)
        x_sequence = x_sequence[:, -1, :]
        # print("x_sequence after LSTM: ", x_sequence)

        # Combine the outputs
        # x_combined = torch.cat((x_initial, x_sequence), dim=1)

        # Fully connected layers
        x_sequence = F.relu(self.fc1(x_sequence))
        # print("x_sequence after fc1: ", x_sequence)
        x_sequence = self.fc2(x_sequence)
        # print("x_sequence after fc2: ", x_sequence)

        # Apply sigmoid for output
        x_sequence = torch.sigmoid(x_sequence)
        # print("x_sequence after sigmoid: ", x_sequence)
        return x_sequence


        # # Fully connected layers
        # x_combined = F.relu(self.fc1(x_combined))
        # x_combined = self.fc2(x_combined)
        
        # #print("X COMBINED before sigmoid: ", x_combined)
        # # Apply sigmoid for output
        # x_combined = torch.sigmoid(x_combined)
        # #print("X_COMBINED after sigmoid: ", x_combined)
        # return x_combined

# class AutoEncoder(nn.Module):



# class CNNLSTM1DWithAttentionStudent(nn.Module):
#     def __init__(self, num_classes=129, lstm_hidden_size=256, lstm_num_layers=2):
#         super(CNNLSTM1DWithAttentionStudent, self).__init__()
#         # Simplifying the initial FC layer
#         self.fc_initial = nn.Linear(5, 32)

#         # Simplifying Conv1d layer after the initial FC
#         self.conv_initial = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)

#         # Additional fully connected layer after Conv1d transformation
#         self.fc_after_conv = nn.Linear(1024, 64)  # Output size reduced

#         # LSTM layer for the sequential part, hidden size and layers reduced
#         self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_hidden_size,
#                             num_layers=lstm_num_layers, batch_first=True,
#                             bidirectional=False)

#         # Simplifying Convolutional and attention layers for the sequence
#         self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.attention = SelfAttention1D(in_channels=16)  # Assuming the implementation of SelfAttention1D is available

#         # Fully connected layers, further simplified
#         self.fc1 = nn.Linear(320, 256)  # Adjust for LSTM output
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         # Split the input
#         x_initial = x[:, :5].unsqueeze(1)  # Adding channel dimension for Conv1d
#         x_sequence = x[:, 5:]

#         # Process the initial data points
#         x_initial = F.relu(self.fc_initial(x_initial))
#         x_initial = F.relu(self.conv_initial(x_initial))
#         x_initial = torch.flatten(x_initial, start_dim=1)  # Flatten to feed into FC
#         x_initial = F.relu(self.fc_after_conv(x_initial))

#         # Process the sequence with Convolutional, Pooling, and Attention
#         x_sequence = x_sequence.unsqueeze(1)  # Adding channel dimension
#         x_sequence = self.pool(F.relu(self.conv1(x_sequence)))
#         x_sequence = self.pool(F.relu(self.conv2(x_sequence)))
#         x_sequence = self.attention(x_sequence)
#         # Reshape for LSTM
#         x_sequence = x_sequence.permute(0, 2, 1)  # [batch_size, seq_length, features]
       
#         # LSTM processing
#         x_sequence, _ = self.lstm(x_sequence)
#         x_sequence = x_sequence[:, -1, :]

#         # Combine the outputs
#         x_combined = torch.cat((x_initial, x_sequence), dim=1)

#         # Fully connected layers
#         x_combined = F.relu(self.fc1(x_combined))
#         x_combined = self.fc2(x_combined)

#         # Apply sigmoid for output
#         x_combined = torch.sigmoid(x_combined)
#         return x_combined

# Ref: https://github.com/fabiozappo/LSTM-Autoencoder-Time-Series/blob/main/code/models/RecurrentAutoencoder.py
class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'ENCODER input dim: {x.shape}')
        # x = x.reshape((batch_size, self.n_features))
        # print(f'ENCODER reshaped dim: {x.shape}')
        x, (_, _) = self.rnn1(x)
        # print(f'ENCODER output rnn1 dim: {x.shape}')
        x, (hidden_n, _) = self.rnn2(x)
        # print(f'ENCODER output rnn2 dim: {x.shape}')
        # print(f'ENCODER hidden_n rnn2 dim: {hidden_n.shape}')
        # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, self.embedding_dim)}')
        # return x.reshape((batch_size, self.embedding_dim))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'DECODER input dim: {x.shape}')
        # x = x.repeat(batch_size, 1) # todo testare se funziona con più feature
        # print(f'DECODER repeat dim: {x.shape}')
        # x = x.reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER reshaped dim: {x.shape}')
        x, (hidden_n, cell_n) = self.rnn1(x)
        # print(f'DECODER output rnn1 dim:/ {x.shape}')
        x, (hidden_n, cell_n) = self.rnn2(x)
        # x = x.reshape((batch_size, self.hidden_dim))
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64, device='cuda', batch_size=32):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, n_features).to(device)

    def forward(self, x):
        # print("AUTOCODER input dimensions in forward: ",x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        # print("AUTOCODER output dimensions in forward: ",x.shape)
        return x
