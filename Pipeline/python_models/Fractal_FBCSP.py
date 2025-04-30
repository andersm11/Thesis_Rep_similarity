from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def petrosian_fd(x):
        x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        N = len(x)
        diff = np.diff(x)
        N_delta = np.sum(diff[:-1] * diff[1:] < 0)
        return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))

class FractalFeatureLayer(nn.Module):
    def __init__(self, input_length, num_scales=5, fd_func=petrosian_fd):
        super().__init__()
        self.input_length = input_length
        self.num_scales = num_scales
        self.fd_func = fd_func
        self.window_sizes = self._compute_window_sizes()
        self.strides = [w // 2 for w in self.window_sizes]  # 50% overlap

    def _compute_window_sizes(self):
        # Compute K window sizes, progressively halved
        return [max(4, int(self.input_length // (2 ** i))) for i in range(self.num_scales)]

    def forward(self, x):
        # x shape: (B, C, T)
        B, C, T = x.shape
        all_fd = []
        length_per_scale = []

        for w, s in zip(self.window_sizes, self.strides):
            n_windows = max(1, (T - w) // s + 1)
            #print("n_windows", n_windows)
            length_per_scale.append(n_windows)
            fd_vals = torch.zeros(B, C, n_windows, device=x.device)

            for b in range(B):
                for c in range(C):
                    signal = x[b, c].cpu().numpy()
                    for i in range(n_windows):
                        start = i * s
                        end = start + w
                        if end > T:
                            break
                        window = signal[start:end]
                        fd = self.fd_func(window)
                        fd_vals[b, c, i] = fd

            all_fd.append(fd_vals)

        # Truncate all fd tensors to the minimum number of windows
        fd_out = torch.cat(all_fd, dim=2) 
        return fd_out


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


    def __init__(self, n_chans, n_outputs, n_times, window_len=124, dropout=0.5, num_scales=5, kernel_size=25, pool_size=20):
        super(ShallowFBCSPNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        #self.fd_layer = FractalFeatureLayer(input_length=window_len, num_scales=num_scales)
        self.spatial = nn.Conv2d(1, 66, (n_chans, 1))
        self.batch_norm = nn.BatchNorm2d(66)
        self.pool = nn.AvgPool2d((1, 10))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1716, n_outputs)

    def forward(self, input):
        B, C,T = input.shape
        x = torch.unsqueeze(input, dim=1)

        #B,K,C,T = x.shape
        #x = self.fd_layer(input)
        #x = torch.unsqueeze(x, dim=1)
        #print(x.shape)
        #x = x.permute(B,1,C,T)
        #print(x.shape)

        x = self.spatial(x)
        x = F.elu(x)
        #print(x.shape)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
