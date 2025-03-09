import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):

        out, _ = self.lstm(x)  # LSTM output shape: [B, n_timesteps, hidden_dim]
        
        return out.permute(0, 2, 1)  # Convert back to [B, hidden_dim, n_timesteps] if needed



##### SPATIAL FØR LSTM, ELLER!!!! CONV TIME I STEDET FOR CONV SPAT!!!!


class ShallowLSTMNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=1, kernel_size=25, pool_size=100, hidden_size=22, nr_layers=1):
        super(ShallowLSTMNet, self).__init__()
        
        
        self.lstm = LSTM(n_chans, hidden_size, nr_layers)  # Independent LSTM for each channel
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (hidden_size, 1))  # Spatial convolution
        self.pool = nn.AvgPool2d((1, pool_size))  # Pooling
        self.batch_norm = nn.BatchNorm2d(num_kernels)  # Batch normalization
        self.dropout = nn.Dropout(dropout)  # Dropout
        self.fc = nn.LazyLinear(n_outputs)  # Fully connected layer
        
    def forward(self, input):
        # input shape: [B, n_channels, n_timesteps, n_features]
        #x = torch.squeeze(input, dim=1)  # Remove the singleton dimension: [B, 1, 22, 1125] → [B, 22, 1125]
        #x = input.permute(0, 2, 1)  # Swap dimensions: [B, 22, 1125] → [B, 1125, 22] (LSTM format)
        #print(input.shape)
        # Apply Independent LSTM for each channel
       #x = torch.unsqueeze(input,dim=-1)
        x=  input.permute(0,2,1)
        #print("lstm input:",x.shape)
        x = self.lstm(x)             
        x = F.elu(x)
        #print("lstm output:",x.shape)
        x = torch.unsqueeze(x,dim=1)
        #print("spat input:",x.shape)
        x = self.spatial(x)
        #x = x.permute(0, 2, 1, 3)  # Change dimensions for Conv2D: [B, n_timesteps, n_channels, hidden_dim]
        
         # Now correctly formatted for LSTM: [B, n_channels, n_timesteps, hidden_dim]
        
        
        # Apply spatial convolution
        x = F.elu(x)
        x = self.batch_norm(x)
        
        # Apply pooling
        x = self.pool(x)
        
        # Apply dropout
        #print("x:",x.shape)
        x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        
        # Squeeze the dimensions and apply the fully connected layer
        #x = torch.squeeze(x, dim=1)  # Remove the channel dimension: [B, n_timesteps, hidden_dim]
        #x = torch.squeeze(x, dim=1)  # Remove the time dimension: [B, hidden_dim]
        
        # Final output: [B, n_outputs]
        x = self.fc(x)
        #print("output:",x.shape)
         
        return x
