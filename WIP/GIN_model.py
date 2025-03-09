from torch.nn import Sequential, Linear, ReLU
import torch
from torch_geometric.nn import GIN, global_add_pool
import torch
import torch.nn as nn
import torch.nn.functional as F


# class GIN(torch.nn.Module):
#     def __init__(self):
#         super(GIN, self).__init__()

#         num_features = 22  # dimension of features for each node. In our case, it's the time-steps
#         dim = 32  # dimension of hidden representations
#         self.dim = dim

#         nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
#         self.conv1 = GINConv(nn1)
#         self.bn1 = torch.nn.BatchNorm1d(dim)

#         nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv2 = GINConv(nn2)
#         self.bn2 = torch.nn.BatchNorm1d(dim)

#         nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv3 = GINConv(nn3)
#         self.bn3 = torch.nn.BatchNorm1d(dim)

#         self.fc1 = Linear(dim, dim)
#         self.fc2 = Linear(dim, 4)

#     def forward(self, x, edge_index=None):

#         #x = x.reshape([-1, 16])
       
#         x = x.permute(0,2,1)
       
#         x = x.reshape([-1, 22])
#         #print(x.shape)
#         x = F.relu(self.conv1(x.float(), edge_index))
#         x = self.bn1(x)
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.bn2(x)
#         x = F.relu(self.conv3(x, edge_index))
#         x = self.bn3(x)
#         total_elements = x.numel()  # This gives the total number of elements

#         batch_size = total_elements // (1125 * self.dim)
#         x = x.view(batch_size, 1125, self.dim)  # (B, 1125, 32)
#         x = x.sum(dim=1)

#         x = F.dropout(x, p=0.3, training=self.training)
#         x = self.fc2(x)
#         return x #F.log_softmax(x, dim=-1)
    

class ShallowGINNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=1, kernel_size=25, pool_size=100, hidden_size=128, nr_layers=2):
        super(ShallowGINNet, self).__init__()
        
        
        self.gin = GIN(n_chans,hidden_size,nr_layers)  # Independent LSTM for each channel
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (64, 1))  # Spatial convolution
        self.pool = nn.AvgPool1d(pool_size)  # Pooling
        self.batch_norm = nn.BatchNorm2d(num_kernels)  # Batch normalization
        self.dropout = nn.Dropout(dropout)  # Dropout
        #self.fc1 = nn.LazyLinear(352)
        self.fc = nn.LazyLinear(n_outputs)  # Fully connected layer
        
    def forward(self, input,edge_index):

        x=  input.permute(0,2,1)
        x = self.gin(x,edge_index)             
       
        x = torch.unsqueeze(x,dim=1)
        
        x = x.permute(0,1,3,2)
        x = self.spatial(x)
        
        
        
        x = F.elu(x)
        x = self.batch_norm(x)
        x = x.reshape(x.size(0), -1)
        x = self.pool(x)
        
        x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)
        return x #F.log_softmax(x)
