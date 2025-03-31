import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowLSTMNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.8, num_kernels=16, kernel_size=10, pool_size=20, hidden_size=20, nr_layers=1):
        super(ShallowLSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(input_size=n_chans, hidden_size=hidden_size, num_layers=nr_layers, batch_first=True, bidirectional=True)
        #self.norm_lstm = nn.LayerNorm(2 * hidden_size)
        self.spatial = nn.Conv1d(in_channels=2 * hidden_size, out_channels=num_kernels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)

        self.batch_norm = nn.BatchNorm1d(num_kernels)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_kernels * (n_times // pool_size), n_outputs)

    def forward(self, input):
        x = input.permute(0, 2, 1)  

        x, _ = self.lstm(x) 

        x = x.permute(0, 2, 1) 

        x = self.spatial(x)  
        x = F.elu(x)

        x = self.batch_norm(x)  

        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  

        x = self.dropout(x)
        x = self.fc(x)  

        return x
