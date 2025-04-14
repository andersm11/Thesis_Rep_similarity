import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import scatter_add,index_add
from CKA_functions import adjacency_matrix_motion
from torch_geometric.nn import SGConv, global_add_pool
import torch_geometric.utils as pyg_utils
    
def check_values(name, tensor):
        if torch.isnan(tensor).any():
            print(f"🚨 NaN detected in {name}!")
        if (tensor.abs() > 1e6).any():
            print(f"⚠️ Exploding value (>1e6) in {name}, Max: {tensor.max().item()}")
        if (tensor.abs() < 1e-6).all():
            print(f"🛑 All values ~0 in {name}")
        if (tensor == 0).any():
            print("⚠️ Warning: Edge weights contain zeros! This may cause NaNs in SGConv.")
        if  (tensor.abs() < 1e-6).all() or (tensor.abs() > 1e6).any() or torch.isnan(tensor).any():
            print(tensor)
        

class SimpleGCNNet(torch.nn.Module):
    def __init__(self, time_steps:int,edge_weights, num_hiddens:int, K=1, dropout=0.5, domain_adaptation=""):
        super(SimpleGCNNet, self).__init__()
        self.edge_weights = nn.Parameter(edge_weights.float(),requires_grad=True)
        self.sgconv = SGConv(time_steps, num_hiddens, K=K, add_self_loops=False)
        self.threshold = 0.1

    def forward(self, x, edge_index, alpha=0):
        mask = self.edge_weights >= self.threshold
        filtered_edge_index = edge_index[:, mask]
        filtered_edge_weights = self.edge_weights[mask]
        x = self.sgconv(x, filtered_edge_index, filtered_edge_weights)
        return x


class ShallowSGCNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, edge_weights, dropout=0.5, num_kernels=20, kernel_size=25, pool_size=20,num_hidden=20,K=2):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.tpool = nn.AvgPool2d((1, 2))
        self.tbatch_norm = nn.BatchNorm2d(num_kernels)
        self.sgconv = SimpleGCNNet(188,edge_weights,num_hidden,K=K)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(640, n_outputs)


    def forward(self, input,edge_index):
        x = torch.unsqueeze(input, dim=1)
        x = self.temporal(x)
        x = F.elu(x)
        x = self.tbatch_norm(x)
        x = self.tpool(x)
        x = self.dropout(x)
        x = self.sgconv(x,edge_index)
        x = F.elu(x)
        print(x.shape)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x