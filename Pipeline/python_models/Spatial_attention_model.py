import torch
from torch import nn
import torch.nn.functional as F

# class SpatialAttention(nn.Module):
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)

#         self.out_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.scale = nn.Parameter(torch.tensor(1.0 / (in_channels ** 0.5)))

#     def forward(self, x):
#         B, temp_out_channels, spat_channels, time_steps = x.shape
#         # Reshape: Merge batch and temp_out_channels -> [B * temp_out_channels, spat_channels, time_steps]
#         x_reshaped = x.view(B * temp_out_channels, spat_channels, time_steps)
#         # Compute queries, keys, values
#         query = self.query(x_reshaped)
#         key = self.key(x_reshaped)
#         value = self.value(x_reshaped)
        

#         # Scaled dot-product attention
#         attn_scores = torch.einsum("bct,bcs->bts", query, key) * self.scale  # [B * temp_out_channels, heads, time_steps, time_steps]
#         attn_weights = torch.softmax(attn_scores, dim=-1)

#         # Apply attention to values
#         out = torch.einsum("bts,bcs->bct", attn_weights, value)

#         # Reshape back: [B * temp_out_channels, spat_channels, time_steps]
#         out = out.reshape(B * temp_out_channels, spat_channels, time_steps)

#         # Apply final transformation
#         out = self.out_conv(out)
#         out = out + x_reshaped  # Residual connection
#         out = out.view(B, temp_out_channels, spat_channels, time_steps)

#         return out

# class SpatialAttention(nn.Module):
#     def __init__(self, spat_channels: int):
#         super().__init__()
#         self.query = nn.Linear(spat_channels, spat_channels)
#         self.key = nn.Linear(spat_channels, spat_channels)
#         self.value = nn.Linear(spat_channels, spat_channels)

#         #self.out_conv = nn.Conv1d(1101, 1101, kernel_size=1, bias=False)
#         self.scale = nn.Parameter(torch.tensor(1.0 / (spat_channels ** 0.5)))

#     def forward(self, x):
#         B, temp_out_channels, spat_channels, time_steps = x.shape
#         x_reshaped = x.view(B * temp_out_channels, time_steps, spat_channels)
#         query = self.query(x_reshaped)
#         key = self.key(x_reshaped)
#         value = self.value(x_reshaped)
#         # print("Q:",query.shape)
#         # print("K:", key.shape)
#         # print("V:", value.shape)

#         attn_scores = torch.einsum("btc,btc->bt", query, key) * self.scale  
#         attn_weights = torch.softmax(attn_scores, dim=-1)

#         out = torch.einsum("bt,btc->btc", attn_weights, value)
#         #out = out.reshape(B * temp_out_channels, spat_channels, time_steps)
#         #print("out:",out.shape)
#         #out = self.out_conv(out)
#         out = out * x_reshaped
#         out = out.view(B, temp_out_channels, spat_channels, time_steps)
#         print(out.shape)
#         return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)

        self.out_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / (in_channels ** 0.5)))

    def forward(self, x):
        B, temp_out_channels, spat_channels, time_steps = x.shape

        x_reshaped = x.view(B * temp_out_channels, spat_channels, time_steps)
        query = self.query(x_reshaped)
        key = self.key(x_reshaped)
        value = self.value(x_reshaped)

        attn_scores = torch.einsum("bct,bcs->bts", query, key) * self.scale  
        attn_weights = torch.softmax(attn_scores, dim=-1)


        out = torch.einsum("bts,bcs->bct", attn_weights, value)
   
        out = out.reshape(B * temp_out_channels, spat_channels, time_steps)
        print("out:",out.shape)
        out = self.out_conv(out)
        out = out + x_reshaped 
        out = out.view(B, temp_out_channels, spat_channels, time_steps)
        return out

class ShallowAttentionNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5, num_kernels=1, kernel_size=25, pool_size=50):
        super(ShallowAttentionNet, self).__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.temporal = nn.Conv2d(1,num_kernels,(1,kernel_size))
        T_out = n_times - kernel_size + 1

        self.spatial_att = TemporalAttention(n_chans)
        

        self.batch_norm = nn.BatchNorm2d(num_kernels) 
        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, input):
        x = torch.unsqueeze(input,dim=1)
        #print("after unsqueeze:", x.shape)
        x = self.temporal(x)
        x = self.spatial_att(x)

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x