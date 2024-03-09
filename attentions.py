import torch
import torch.nn as nn
import torch.nn.functional as F



device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiHeadAttention1D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadAttention1D, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels

        # Ensure the in_channels is divisible by the number of heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.depth = in_channels // num_heads

        # Define linear layers for Q, K, V transformations for each head
        self.query = nn.Linear(37, in_channels)
        self.key = nn.Linear(37, in_channels)
        self.value = nn.Linear(37, in_channels)

        # Linear layer for the concatenated outputs
        self.dense = nn.Linear(in_channels, in_channels)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.shape[0]

        # Generate query, key, and value for all heads
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Split for multi-head attention
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        scale = torch.sqrt(torch.FloatTensor([self.depth])).to(x.device)
        scaled_attention_logits = matmul_qk / scale

        # Softmax to get probabilities
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Apply attention to the value
        output = torch.matmul(attention_weights, value)

        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.in_channels)

        # Final linear layer
        output = self.dense(output)

        return output




class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention1D, self).__init__()
        # Query, Key, Value transformations
        self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1).to(device)
        self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1).to(device)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1).to(device)

        # Scale factor for the attention scores
        self.scale = torch.sqrt(torch.FloatTensor([in_channels])).to(device)

    def forward(self, x):
        # Generate query, key, value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores
        scores = torch.matmul(query.transpose(1, 2), key) / self.scale

        # Apply softmax to get probabilities
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to the value
        attended = torch.matmul(attention_weights, value.transpose(1, 2))
        attended = attended.transpose(1, 2)

        # Combine attended values with the input (Optional: could be just 'attended')
        out = x + attended

        return out



