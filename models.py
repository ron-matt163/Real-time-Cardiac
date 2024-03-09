import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F

from attentions import *

class CNNLSTM1DWithAttentionTeacher(nn.Module):
    def __init__(self, num_classes=129, lstm_hidden_size=512, lstm_num_layers=3):
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
        self.fc1 = nn.Linear(lstm_hidden_size*2 + 128, 512)  # Adjust the input size for combined output
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Split the input
        x_initial = x[:, :5].unsqueeze(1)  # Adding channel dimension for Conv1d
        x_sequence = x[:, 5:]

        # Process the initial data points
        x_initial = F.relu(self.fc_initial(x_initial))
        x_initial = F.relu(self.conv_initial(x_initial))
        x_initial = torch.flatten(x_initial, start_dim=1)  # Flatten to feed into FC
        x_initial = F.relu(self.fc_after_conv(x_initial))

        # Process the sequence with Convolutional, Pooling, and Attention
        x_sequence = x_sequence.unsqueeze(1)  # Adding channel dimension
        x_sequence = self.pool(F.relu(self.conv1(x_sequence)))
        x_sequence = self.pool(F.relu(self.conv2(x_sequence)))
        x_sequence = self.pool(F.relu(self.conv3(x_sequence)))
        x_sequence = self.attention(x_sequence)
        # print(x_sequence.shape)
        # Reshape for LSTM
        x_sequence = x_sequence.permute(0, 2, 1)  # [batch_size, seq_length, features]

        # LSTM processing
        x_sequence, _ = self.lstm(x_sequence)
        x_sequence = x_sequence[:, -1, :]

        # Combine the outputs
        x_combined = torch.cat((x_initial, x_sequence), dim=1)

        # Fully connected layers
        x_combined = F.relu(self.fc1(x_combined))
        x_combined = self.fc2(x_combined)

        # Apply sigmoid for output
        x_combined = torch.sigmoid(x_combined)
        return x_combined


class CNNLSTM1DWithAttentionStudent(nn.Module):
    def __init__(self, num_classes=129, lstm_hidden_size=256, lstm_num_layers=2):
        super(CNNLSTM1DWithAttentionStudent, self).__init__()
        # Simplifying the initial FC layer
        self.fc_initial = nn.Linear(5, 32)

        # Simplifying Conv1d layer after the initial FC
        self.conv_initial = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)

        # Additional fully connected layer after Conv1d transformation
        self.fc_after_conv = nn.Linear(1024, 64)  # Output size reduced

        # LSTM layer for the sequential part, hidden size and layers reduced
        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True,
                            bidirectional=False)

        # Simplifying Convolutional and attention layers for the sequence
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.attention = SelfAttention1D(in_channels=16)  # Assuming the implementation of SelfAttention1D is available

        # Fully connected layers, further simplified
        self.fc1 = nn.Linear(320, 256)  # Adjust for LSTM output
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Split the input
        x_initial = x[:, :5].unsqueeze(1)  # Adding channel dimension for Conv1d
        x_sequence = x[:, 5:]

        # Process the initial data points
        x_initial = F.relu(self.fc_initial(x_initial))
        x_initial = F.relu(self.conv_initial(x_initial))
        x_initial = torch.flatten(x_initial, start_dim=1)  # Flatten to feed into FC
        x_initial = F.relu(self.fc_after_conv(x_initial))

        # Process the sequence with Convolutional, Pooling, and Attention
        x_sequence = x_sequence.unsqueeze(1)  # Adding channel dimension
        x_sequence = self.pool(F.relu(self.conv1(x_sequence)))
        x_sequence = self.pool(F.relu(self.conv2(x_sequence)))
        x_sequence = self.attention(x_sequence)
        # Reshape for LSTM
        x_sequence = x_sequence.permute(0, 2, 1)  # [batch_size, seq_length, features]
       
        # LSTM processing
        x_sequence, _ = self.lstm(x_sequence)
        x_sequence = x_sequence[:, -1, :]

        # Combine the outputs
        x_combined = torch.cat((x_initial, x_sequence), dim=1)

        # Fully connected layers
        x_combined = F.relu(self.fc1(x_combined))
        x_combined = self.fc2(x_combined)

        # Apply sigmoid for output
        x_combined = torch.sigmoid(x_combined)
        return x_combined
