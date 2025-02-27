import torch
from torch import nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
       # self.fc_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [B, temp_out_channels, spat_channels, time_steps]
        B, temp_out_channels, spat_channels, time_steps = x.shape
        # Compute queries, keys, and values
        x_reshaped = x.view(B * temp_out_channels, spat_channels, time_steps)
        query = self.query(x_reshaped)  # [B, temp_out_channels, spat_channels, time_steps]
        key = self.key(x_reshaped)      # [B, temp_out_channels, spat_channels, time_steps]
        value = self.value(x_reshaped)  # [B, temp_out_channels, spat_channels, time_steps]
        attention_scores = torch.matmul(query.transpose(1, 2), key)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, value.transpose(1, 2))  # [B * temp_out_channels, time_steps, spat_channels]
        output = output.transpose(1, 2)  # [B * temp_out_channels, spat_channels, time_steps]

        # Reshape back to [B, temp_out_channels, spat_channels, time_steps]
        output = output.view(B, temp_out_channels, spat_channels, time_steps)
        
        # Apply final fully connected layer (if needed)
        #out = self.fc_out(out)
        return output




class ShallowAttentionNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=10, kernel_size=25, pool_size=100):
        super(ShallowAttentionNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1,num_kernels,(1,kernel_size))
        # **Spatial Attention Module (Replaces Temporal & Spatial Conv)**
        self.spatial_attention = SpatialAttention(in_channels=n_chans)

        self.batch_norm = nn.BatchNorm2d(num_kernels) 
        #self.minipool = nn.MaxPool2d((1,self.minipool_size))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, input):
        x = torch.unsqueeze(input, dim=1)  # Add channel dimension [B, 1, C, T]
        #print("after unsqueeze:", x.shape)
        x = self.temporal(x)
        x = self.spatial_attention(x)  


        x = F.elu(x)
        #print("after att:", x.shape)
        x = self.batch_norm(x)
        #print("after batch norm:", x.shape)
        x = self.pool(x)
        #print("after pool",x.shape)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x