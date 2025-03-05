import torch
import torch.nn as nn
import torch.nn.functional as F

class IndependentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(IndependentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: [B, n_channels, n_timesteps, input_dim]
        B, n_channels, n_timesteps = x.size()

        # Create an empty list to store LSTM outputs for each channel
        outputs = []

        for i in range(n_channels):
            #print(x.shape)
            channel_input = x[:, i, :]  # Extract the i-th channel (shape: [B, n_timesteps, input_dim])
            channel_input = torch.unsqueeze(channel_input,1)
            channel_input = channel_input.permute(0,2,1)

            #print("chan input",channel_input.shape)
            out, _ = self.lstm(channel_input)  # LSTM output (shape: [B, n_timesteps, hidden_dim])
            #print("out shape:",out.shape)
            outputs.append(out)
        
        # Stack the outputs to form the final shape: [B, n_channels, n_timesteps, hidden_dim]
        final_output = torch.stack(outputs, dim=1)
        final_output = final_output.permute(0,1,3,2)
        #print("lstm output:",final_output.shape)
        
        return final_output



##### SPATIAL FØR LSTM, ELLER!!!! CONV TIME I STEDET FOR CONV SPAT!!!!


class ShallowLSTMNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=22, kernel_size=25, pool_size=100, hidden_size=22, nr_layers=1):
        super(ShallowLSTMNet, self).__init__()
        
        self.lstm = IndependentLSTM(n_times, hidden_size, nr_layers)  # Independent LSTM for each channel
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
        x = self.lstm(input)  # Now correctly formatted for LSTM: [B, n_channels, n_timesteps, hidden_dim]
        
        x = x.permute(0, 2, 1, 3)  # Change dimensions for Conv2D: [B, n_timesteps, n_channels, hidden_dim]
        
        # Apply spatial convolution
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        
        # Apply pooling
        x = self.pool(x)
        
        # Apply dropout
        #print("x:",x.shape)
        x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        
        # Squeeze the dimensions and apply the fully connected layer
        x = torch.squeeze(x, dim=1)  # Remove the channel dimension: [B, n_timesteps, hidden_dim]
        x = torch.squeeze(x, dim=1)  # Remove the time dimension: [B, hidden_dim]
        
        # Final output: [B, n_outputs]
        x = self.fc(x)
        #print("output:",x.shape)
         
        return x
