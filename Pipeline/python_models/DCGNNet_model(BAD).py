import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support

def normalize_A(A,lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        # print("out:",out.shape)
        # print("self.weight: ",self.weight.shape)
        out = out.view(-1, self.in_channels) 
        out = torch.matmul(out, self.weight)
        out =  out.view(x.size(0),x.size(1),x.size(2),x.size(3))
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        
        for i in range(len(self.gc)):
            if i == 0:
                # print(self.gc[i])
                # print("x:",x.shape)
                # print("adj i: ",adj[i].shape)
                # print("adj:",adj)
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        
        return result

class ShallowGNNNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, k_adj, cheb_out_channels =10, dropout=0.8, num_kernels=10, kernel_size=25, pool_size=50):
        super(ShallowGNNNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        self.K = k_adj
        self.A = nn.Parameter(torch.FloatTensor(n_chans,n_chans).cuda())
        
        self.cheby = Chebynet(num_kernels, k_adj, cheb_out_channels)
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4840 , n_outputs)

    def forward(self, input):
        x = torch.unsqueeze(input, dim=1)
        x = self.temporal(x)
        L = normalize_A(self.A)
        x = self.cheby(x,L)
        x = F.elu(x)
       
        #x = x.permute(1,0)
        # print(x.shape)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc(x)
        return x
