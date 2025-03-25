import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNN(nn.Module):
    def __init__(self,n_chans,num_kernels,device):
        super(GNN, self).__init__()
        self.gcnConv = GCNConv(num_kernels, num_kernels)  
        self.adj_matrix =  torch.ones(n_chans, n_chans)  # Initialized randomly
        self.edge_index = self.create_fully_connected_edges(n_chans).to(device)
        self.edge_weights = nn.Parameter(torch.rand(self.adj_matrix.numel(), requires_grad=True, device=device))


    def create_fully_connected_edges(self, n_chans):
        """Creates a fully connected graph edge index."""
        row, col = torch.meshgrid(torch.arange(n_chans), torch.arange(n_chans), indexing='ij')
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)  # Shape: [2, num_edges]
        return edge_index
    
    def forward(self, input):
        print(self.edge_weights)
        x = self.gcnConv(input, self.edge_index, self.edge_weights) 
        return x

class ShallowGNNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.8, num_kernels=10, kernel_size=25, pool_size=100,device='cpu'):
        super(ShallowGNNNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.num_kernels = num_kernels
        
        # Temporal convolution
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.gnn = GNN(n_chans,num_kernels,device)
        
        self.batch_norm = nn.BatchNorm2d(n_chans)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)

        temp_out = (n_times - kernel_size + 1)  
        pool_out = temp_out // pool_size
        input_size = num_kernels * n_chans * pool_out  
        self.fc = nn.Linear(input_size, n_outputs)

    

    def forward(self, input):
        # Add batch dimension
        x = torch.unsqueeze(input, dim=1)
        
        # Temporal convolution (output shape: [B, num_kernels, C, T])
        x = self.temporal(x)
        B, num_kernels, C, T = x.shape
        x = x.view(B, num_kernels, C*T).transpose(1, 2) 
        x = x.contiguous().view(B * C * T, num_kernels) 

        x = self.gnn(x)  
        x = x.view(B, num_kernels, C, T).transpose(1, 2)  # Reshape to match [B, num_kernels, C, T]

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

    
