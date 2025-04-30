import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import scatter_add,index_add
from CKA_functions import adjacency_matrix_motion,adjacency_matrix_distance_motion
from torch_geometric.nn import SGConv, global_add_pool
import torch_geometric.utils as pyg_utils
        

class SimpleGCNNet(torch.nn.Module):
    def __init__(self, time_steps:int, edge_weights, num_hiddens:int, K=1):
        super(SimpleGCNNet, self).__init__()
        self.edge_weights = nn.Parameter(edge_weights.float(),requires_grad=True)
        self.sgconv = SGConv(time_steps, num_hiddens, K=K, add_self_loops=True)
            
    def forward(self, x, edge_index):
        #self.edge_weights.data[F.elu(self.edge_weights.data)  <=  0] = self.epsilon 
       # B,K,C,T= x.shape
       # x = x.view(B, C, K * T) 
        #filtered_weights = F.softplus(self.edge_weights)
        x = F.normalize(x, p=2, dim=-1)
        x = self.sgconv(x, edge_index, self.edge_weights)
        return x

class FractalSGCNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, fd_feat_len, edge_weights, dropout=0.5, num_hidden=50, pool_size=10, K=1):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.fd_feat_len = fd_feat_len  # Number of time-like features from FD
        self.sgconv = SimpleGCNNet(fd_feat_len, edge_weights, num_hidden, K=K)

        self.batch_norm = nn.BatchNorm2d(1)  # Normalize across (1, C, T_fd)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)

        # Dynamically compute linear layer input size
        dummy_input = torch.zeros(1, n_chans, fd_feat_len)  # (B, C, T_fd)
        self.fc = nn.Linear(self._get_flattened_size(dummy_input), n_outputs)

    def _get_flattened_size(self, x):
        with torch.no_grad():
            adj_m, pos = adjacency_matrix_motion()
            adj_dis_m, dm = adjacency_matrix_distance_motion(pos, delta=6)
            edge_index = self.get_e_index(dm)
            x = self.sgconv(x, edge_index)
            x = F.elu(x)
            x = x.unsqueeze(1)  # (B, 1, C, T_fd)
            x = self.batch_norm(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return x.size(1)

    def get_e_index(self, dm):
        threshold = 0
        source_nodes = []
        target_nodes = []
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                if dm[i, j] >= threshold:
                    source_nodes.append(i)
                    target_nodes.append(j)
        return torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    def forward(self, x, edge_index):
        # x: (B, C, T_fd)
        x = self.sgconv(x, edge_index)        # (B, C, T_fd)
        x = F.elu(x)
        x = x.unsqueeze(1)                    # (B, 1, C, T_fd)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)             # flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x