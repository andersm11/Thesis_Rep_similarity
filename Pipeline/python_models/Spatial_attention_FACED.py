import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention_matrix(x, att_unsq, batch_idx=0, kernel_idx=0):
    # Extract the attention map for the specific batch and kernel
    attention_map = att_unsq[batch_idx, kernel_idx, :, :].cpu().detach().numpy()  # Shape: [C, T]

    # Extract the raw input data for the specific batch and kernel
    input_data = x[batch_idx, kernel_idx, :, :].cpu().detach().numpy()  # Shape: [C, T]

    # Plot the Attention Map as a Heatmap
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(attention_map, cmap='viridis', cbar=True, xticklabels=range(attention_map.shape[1]),
                yticklabels=range(attention_map.shape[0]))
    plt.title(f'Spatial Attention Map (Batch {batch_idx}, Kernel {kernel_idx})')
    plt.xlabel('Time Steps (T)')
    plt.ylabel('Electrodes (C)')

    # Plot the Input Data (Before Attention) as a Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(input_data, cmap='coolwarm', cbar=True, xticklabels=range(input_data.shape[1]),
                yticklabels=range(input_data.shape[0]))
    plt.title(f'Input Data (Batch {batch_idx}, Kernel {kernel_idx})')
    plt.xlabel('Time Steps (T)')
    plt.ylabel('Electrodes (C)')

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Compute and Plot the Output (After Applying Attention) as a Heatmap
    output_data = input_data * attention_map  # Element-wise multiplication (Shape: [C, T])

    plt.figure(figsize=(8, 6))
    sns.heatmap(output_data, cmap='coolwarm', cbar=True, xticklabels=range(output_data.shape[1]),
                yticklabels=range(output_data.shape[0]))
    plt.title(f'Output Data After Applying Attention (Batch {batch_idx}, Kernel {kernel_idx})')
    plt.xlabel('Time Steps (T)')
    plt.ylabel('Electrodes (C)')
    plt.show()
    
def visualize_difference(x, out, batch_idx=0, kernel_idx=0):
    # Extract the raw input data and output data for the specific batch and kernel
    input_data = x[batch_idx, kernel_idx, :, :].cpu().detach().numpy()  # Shape: [C, T]
    output_data = out[batch_idx, kernel_idx, :, :].cpu().detach().numpy()  # Shape: [C, T]
    
    # Calculate the difference between input and output
    difference = output_data - input_data

    # Plot the Difference Matrix as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(difference, cmap='coolwarm', cbar=True, xticklabels=range(difference.shape[1]),
                yticklabels=range(difference.shape[0]))
    plt.title(f'Difference Between Input and Output (Batch {batch_idx}, Kernel {kernel_idx})')
    plt.xlabel('Time Steps (T)')
    plt.ylabel('Electrodes (C)')
    plt.show()


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, pool_size: int, num_heads: int = 2):  
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False)

        # Pooling to Reduce Dimensionality
        self.pool = nn.MaxPool2d((1, pool_size))

    def forward(self, x):
        B, K, C, T = x.shape  
        attention_map = self.conv(x) 
 
        attention_map = attention_map.view(-1, x.size(2), x.size(3))  # Shape: [B*K, C, T]
        # **Global Average Pooling over Time**
        attention_map =  torch.sigmoid(attention_map)
        attention_map_avg = attention_map.mean(dim=2)  
        attention_map_avg = attention_map_avg.view(x.size(0), x.size(1), x.size(2))  
        #print(attention_map_avg.shape)
        attention_map = attention_map_avg.unsqueeze(-1)  # (B, K, C, 1)
       #print(attention_map.shape)
        x = x * attention_map

        # **Apply Pooling**
        x = self.pool(x)

        return x  # Weighted Feature Maps

class ShallowAttentionNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=50):
        super(ShallowAttentionNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))  # Reduce num_kernels to 5
        self.spatial_att = SpatialAttention(num_kernels,pool_size//10)

        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)

        # **Major change: Reduce FC layer complexity**
        reduced_size = num_kernels * n_chans * ((n_times - kernel_size + 1) // (pool_size*8))
        self.fc = nn.Linear(1280 , n_outputs)  # No dependence on n_chans

    def forward(self, input):
        x = torch.unsqueeze(input, dim=1)
        x = self.temporal(x)
        x = F.elu(x)
        x = self.spatial_att(x)

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)

       # x = x.mean(dim=2)  # **Global Average Pooling across electrodes**
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
       # print(x.shape)
        x = self.fc(x)
        return x