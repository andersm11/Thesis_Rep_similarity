import torch.nn as nn
import torch
from torch_geometric.nn import SGConv
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
    elif isinstance(m, SGConv):  # Initialize GCN layers
        torch.nn.init.xavier_uniform_(m.lin.weight)  
        if m.lin.bias is not None:
            torch.nn.init.zeros_(m.lin.bias)

