from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


class ShallowFBCSPNet(nn.Module):
    """An implementation of the ShallowFBCSPNet model from https://arxiv.org/abs/1703.05051 

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        timepoints (int, optional): Number of timepoints in the input data. Default is 1000.
        dropout (float, optional): Dropout probability. Default is 0.5.
        num_kernels (int, optional): Number of convolutional kernels. Default is 40.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 25.
        pool_size (int, optional): Size of the pooling window. Default is 100.
    """
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=20, kernel_size=25, pool_size=50):
        super(ShallowFBCSPNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (n_chans, 1))
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)
        dummy_input = torch.zeros(1, 1, self.n_chans, self.n_times, device=next(self.parameters()).device)
        feature_size = self._get_flattened_size(dummy_input)
        self.fc = nn.Linear(feature_size, n_outputs)
        
    def _get_flattened_size(self, x):
        with torch.no_grad():
            x = self.temporal(x)
            x = F.elu(x)
            #x = self.tpool(x)
            #x = self.dropout(x)
            x = self.spatial(x)
            x = F.elu(x)
            x = self.batch_norm(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return x.size(1)


    def forward(self, input):
        x = torch.unsqueeze(input, dim=1)
        x = self.temporal(x)
        x = F.elu(x)
        #x = self.tpool(x)
        x = self.dropout(x)
        x = self.spatial(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
   