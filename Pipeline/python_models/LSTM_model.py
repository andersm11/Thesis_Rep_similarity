import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowLSTMNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.7, num_kernels=10, kernel_size=10, pool_size=20, hidden_size=22, nr_layers=1):
        super(ShallowLSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(input_size=n_chans, hidden_size=hidden_size, num_layers=nr_layers, batch_first=True, bidirectional=False)
        self.spatial = nn.Conv2d(1, num_kernels*2, (n_chans, 1))
        self.batch_norm = nn.BatchNorm2d(num_kernels*2)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_kernels*2 * (n_times // pool_size), n_outputs)

    def forward(self, input):
        x = input.permute(0, 2, 1)  
        x, _ = self.lstm(x) 
        x = x.permute(0, 2, 1) 
        x = torch.unsqueeze(x, dim=1)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
