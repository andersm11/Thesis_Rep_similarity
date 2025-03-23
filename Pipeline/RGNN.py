import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import scatter_add,index_add
from CKA_functions import adjacency_matrix_motion
from torch_geometric.nn import SGConv, global_add_pool


class SimpleGCNNet(torch.nn.Module):
    def __init__(self, time_steps:int, edge_weights, num_hiddens:int, K=1, dropout=0.5, domain_adaptation=""):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of motion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SimpleGCNNet, self).__init__()
        self.edge_weights = nn.Parameter(edge_weights.float())
        self.rgnn = SGConv(time_steps,num_hiddens,K=K)
        #self.dropout = nn.Dropout(dropout)
            
    def forward(self, x,edge_index, alpha=0):
        #batch_size,K,C,T = x.shape
        x = self.rgnn(x, edge_index,self.edge_weights)
       # x = self.dropout(x)
        return x
    
    
class ShallowRGNNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, edge_weights, dropout=0.7, num_kernels=10, kernel_size=25, pool_size=20,num_hidden=20):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.rgnn = SimpleGCNNet(55,edge_weights,num_hidden)
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4400, n_outputs)


    def forward(self, input,edge_index):
        x = torch.unsqueeze(input, dim=1)
        x = self.temporal(x)
        
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = self.rgnn(x,edge_index)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x