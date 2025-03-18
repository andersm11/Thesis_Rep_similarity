import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RNN, self).__init__()
        self.RNN = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.RNN(x)
        return out.permute(0, 2, 1) 


class ShallowRNNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.7, num_kernels=10, kernel_size=25, pool_size=25, hidden_size=64, nr_layers=1):
        super(ShallowRNNNet, self).__init__()
        
        self.RNN = RNN(n_chans, hidden_size, nr_layers)  
        self.spatial = nn.Conv2d(1, num_kernels, (hidden_size, 1)) 
        self.pool = nn.AvgPool2d((1, pool_size)) 
        self.batch_norm = nn.BatchNorm2d(num_kernels)  
        self.dropout = nn.Dropout(dropout)  
        self.fc = nn.LazyLinear(n_outputs) 
        
    def forward(self, input):
        x = input.permute(0, 2, 1)  
        
        x = self.RNN(x) 
        x = F.elu(x)
        x = torch.unsqueeze(x, dim=1)  
        
        x = self.spatial(x) 
        x = F.elu(x)  
        x = self.batch_norm(x) 
        
        x = self.pool(x)  
        
        x = x.reshape(x.size(0), -1)  
        
        x = self.dropout(x)  
        
        x = self.fc(x)  
        return x