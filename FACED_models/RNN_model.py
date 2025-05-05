import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowRNNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=50, hidden_size=32, nr_layers=1):
        super(ShallowRNNNet, self).__init__()
        self.RNN = nn.RNN(input_size=n_chans, hidden_size=hidden_size, num_layers=nr_layers, batch_first=True)
        self.spatial = nn.Conv2d(1, num_kernels*2, (n_chans, 1))
        self.batch_norm = nn.BatchNorm2d(num_kernels*2)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_chans, n_times)  # (B, C, T)
            x = dummy_input.permute(0, 2, 1)                # (B, T, C)
            x, _ = self.RNN(x)                              # (B, T, H)
            x = x.permute(0, 2, 1).unsqueeze(1)             # (B, 1, H, T)
            x = self.spatial(x)                             # (B, F, 1, T)
            x = F.elu(x)
            x = self.batch_norm(x)
            x = self.pool(x)                                # (B, F, 1, T//pool_size)
            x = x.view(x.size(0), -1)                       # (B, F * T//pool_size)
            fc_input_dim = x.shape[1]

        self.fc = nn.Linear(fc_input_dim, n_outputs)
        
    def forward(self, input):
        x = input.permute(0, 2, 1)  
        x,_ = self.RNN(x) 
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