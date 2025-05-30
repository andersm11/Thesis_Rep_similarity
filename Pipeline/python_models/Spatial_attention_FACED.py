import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        B, K, C, T = x.shape  
        attention_map = self.conv(x) 
        attention_map = attention_map.view(-1, x.size(2), x.size(3)) 
        attention_map =  torch.sigmoid(attention_map)
        attention_map_avg = attention_map.mean(dim=2)  
        attention_map_avg = attention_map_avg.view(x.size(0), x.size(1), x.size(2))  
        attention_map = attention_map_avg.unsqueeze(-1) 
        x = x * attention_map
        return x

class ShallowAttentionNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=20, kernel_size=25, pool_size=100):
        super(ShallowAttentionNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))  
        self.spatial_att = SpatialAttention(num_kernels)
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        dummy_input = torch.zeros(1, 1, self.n_chans, self.n_times, device=next(self.parameters()).device)
        feature_size = self._get_flattened_size(dummy_input)
        self.fc = nn.Linear(feature_size , n_outputs)  
        
    def _get_flattened_size(self, x):
        with torch.no_grad():
            x = self.temporal(x)
            x = F.elu(x)
            x = self.dropout(x)
            x = self.spatial_att(x)
            x = F.elu(x)
            x = self.batch_norm(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return x.size(1)

    def forward(self, input):
        x = torch.unsqueeze(input, dim=1)
        x = self.temporal(x)
        x = F.elu(x)
        x = self.spatial_att(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x