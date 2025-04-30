import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import scatter_add,index_add
from CKA_functions import adjacency_matrix_motion
from torch_geometric.nn import SGConv, global_add_pool
import torch_geometric.utils as pyg_utils
from CKA_functions import adjacency_matrix_FACED,adjacency_matrix_distance_FACED

    
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
    def __init__(self, time_steps:int,edge_weights, num_hiddens:int, K=1):
        super(SimpleGCNNet, self).__init__()
        self.edge_weights = nn.Parameter(edge_weights.float(),requires_grad=True)
        self.sgconv = SGConv(time_steps, num_hiddens, K=K, add_self_loops=False)
        self.threshold = 0.1

    def forward(self, x, edge_index, alpha=0):
        #mask = self.edge_weights >= self.threshold
        #filtered_edge_index = edge_index[:, mask]
        #filtered_edge_weights = self.edge_weights[mask]
        #filtered_weights = F.softplus(self.edge_weights)
        x = F.normalize(x, p=2, dim=-1)  # Normalize each node's features to unit norm
        x = self.sgconv(x, edge_index, self.edge_weights)
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
        gcn_input_dim = self._get_gcn_input_dim(n_chans, n_times)
        self.sgconv = SimpleGCNNet(gcn_input_dim, edge_weights, num_hidden, K=K)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        dummy_input = torch.zeros(1, 1, n_chans, n_times)
        feature_size = self._get_flattened_size(dummy_input)
        self.fc = nn.Linear(feature_size, n_outputs)
        
    def _get_gcn_input_dim(self, n_chans, n_times):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_chans, self.n_times, device=next(self.parameters()).device)
            x = self.temporal(dummy_input)  
            x = F.elu(x)
            x = self.tbatch_norm(x)
            x = self.tpool(x)  
            return x.size(-1)  

    def get_e_index(self,dm):
        threshold = 0  # Adjust as needed

        source_nodes = []
        target_nodes = []

        # Iterate over all elements in the distance matrix, including self-loops and duplicates
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):  # Iterate over all pairs, including (i, i)
                if dm[i, j] >= threshold:  # If the distance meets the condition
                    source_nodes.append(i)  # Source node
                    target_nodes.append(j)  # Target node

        # Create the edge_index tensor
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        return edge_index
    
    def _get_flattened_size(self, x):
        with torch.no_grad():
            x = self.temporal(x)
            x = F.elu(x)
            x = self.tbatch_norm(x)
            x = self.tpool(x)
            x = self.dropout(x)
            adj_m,pos = adjacency_matrix_FACED()
            #print(adj_m)
            adj_dis_m, dm = adjacency_matrix_distance_FACED(pos,delta=6)
            edge_index = self.get_e_index(dm)
            x = self.sgconv(x,edge_index)
            x = F.elu(x)
            x = self.batch_norm(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return x.size(1)

    def forward(self, input,edge_index):
        x = torch.unsqueeze(input, dim=1)
        x = self.temporal(x)
        x = F.elu(x)
        x = self.tbatch_norm(x)
        x = self.tpool(x)
        #x = self.dropout(x)
        x = self.sgconv(x,edge_index)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x