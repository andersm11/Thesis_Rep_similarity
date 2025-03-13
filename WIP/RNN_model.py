import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RNN, self).__init__()
        self.RNN = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.RNN(x)  # Correct reference: Use self.RNN here
        return out.permute(0, 2, 1)  # Convert back to [B, hidden_dim, n_timesteps] if needed


class ShallowRNNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.7, num_kernels=1, kernel_size=25, pool_size=200, hidden_size=64, nr_layers=2):
        super(ShallowRNNNet, self).__init__()
        
        self.RNN = RNN(n_chans, hidden_size, nr_layers)  # Independent RNN for each channel
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (hidden_size, 1))  # Spatial convolution
        self.pool = nn.AvgPool2d((1, pool_size))  # Pooling
        self.batch_norm = nn.BatchNorm2d(num_kernels)  # Batch normalization
        self.dropout = nn.Dropout(dropout)  # Dropout
        self.fc = nn.LazyLinear(n_outputs)  # Fully connected layer
        
    def forward(self, input):
        # input shape: [B, n_channels, n_timesteps, n_features]
        x = input.permute(0, 2, 1)  # Swap dimensions: [B, n_channels, n_timesteps, n_features] → [B, n_timesteps, n_channels]
        
        x = self.RNN(x)  # Use self.RNN here, correct reference
        x = F.elu(x)  # Apply ELU activation
        x = torch.unsqueeze(x, dim=1)  # Add a new dimension to match Conv2D input
        
        x = self.spatial(x)  # Apply spatial convolution
        x = F.elu(x)  # Apply ELU activation
        x = self.batch_norm(x)  # Batch normalization
        
        x = self.pool(x)  # Apply pooling
        
        x = x.reshape(x.size(0), -1)  # Flatten the output
        
        x = self.dropout(x)  # Apply dropout
        
        x = self.fc(x)  # Apply fully connected layer for final output
        return x