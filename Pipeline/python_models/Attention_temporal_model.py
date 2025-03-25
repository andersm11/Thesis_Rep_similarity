from torch import nn
import torch
import torch.nn.functional as F



# # def spatial_attention(input_feature):
# #     kernel_size = 7
    
# #     if K.image_data_format() == "channels_first":
# #         channel = input_feature.shape[1]
# #         cbam_feature = Permute((2,3,1))(input_feature)
# #     else:
# #         channel = input_feature.shape[-1]
# #         cbam_feature = input_feature
    
# #     avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
# #     assert avg_pool.shape[-1] == 1
# #     max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
# #     assert max_pool.shape[-1] == 1
# #     concat = Concatenate(axis=3)([avg_pool, max_pool])
# #     assert concat.shape[-1] == 2
# #     cbam_feature = Conv2D(filters = 1,
# #                     kernel_size=kernel_size,
# #                     strides=1,
# #                     padding='same',
# #                     activation='sigmoid',
# #                     kernel_initializer='he_normal',
# #                     use_bias=False)(concat)    
# #     assert cbam_feature.shape[-1] == 1
    
# #     if K.image_data_format() == "channels_first":
# #         cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
# #     return multiply([input_feature, cbam_feature])

# class TemporalAttention(nn.Module):
#     def __init__(self, time_steps,n_dropout = 0.9):
#         super().__init__()
#         self.query = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)
#         self.key = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)
#         self.value = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)

#         self.out_conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
#         self.dropout = nn.Dropout(n_dropout)
#         self.scale = nn.Parameter(torch.tensor(1.0 / (time_steps ** 0.5)))
        

#     def forward(self, x):
#         B, temp_out_channels, spat_channels, time_steps = x.shape
#         x_reshaped = x.view(B * temp_out_channels, time_steps, spat_channels)
#         print(x_reshaped.shape)
#         query = self.query(x_reshaped)
#         key = self.key(x_reshaped)     
#         value = self.value(x_reshaped)  
#         print("Q:",query.shape)
#         print("K:", key.shape)

#         attn_scores = torch.einsum("bc,bc->bc", query, key) * self.scale  
#         attn_weights = torch.softmax(attn_scores, dim=-1)  

#         out = torch.einsum("bc,bc->bc", attn_weights, value)  
#         print(out.shape)
#         out = self.out_conv(out)
#         out = self.dropout(out)
#         out = out + x_reshaped  
#         out = out.view(B, temp_out_channels, spat_channels, time_steps)

#         return out


# class TemporalAttention(nn.Module):
#     def __init__(self, time_steps,n_dropout = 0.9):
#         super().__init__()
#         self.query = nn.Linear(time_steps, time_steps)
#         self.key = nn.Linear(time_steps, time_steps)
#         self.value = nn.Linear(time_steps, time_steps)

#         self.out_conv = nn.Conv1d(22, 22, kernel_size=1, bias=False)
#         #self.dropout = nn.Dropout(n_dropout)
#         self.scale = nn.Parameter(torch.tensor(1.0 / (time_steps ** 0.5)))
        

#     def forward(self, x):
#         B, temp_out_channels, spat_channels, time_steps = x.shape
#         x_reshaped = x.view(B * temp_out_channels, spat_channels, time_steps)
#         print(x_reshaped.shape)
#         query = self.query(x_reshaped)
#         key = self.key(x_reshaped)     
#         value = self.value(x_reshaped)  
#         print("Q:",query.shape)
#         print("K:", key.shape)
#         print("V:", value.shape)

#         attn_scores = torch.einsum("bct,bct->bc", query, key) * self.scale  
#         attn_weights = torch.softmax(attn_scores, dim=-1)  

#         out = torch.einsum("bc,bct->bct", attn_weights, value)  
#         print("out:",out.shape)
#         out = self.out_conv(out)
#         #out = self.dropout(out)
#         #out = out + x_reshaped  
#         out = out.view(B, temp_out_channels, spat_channels, time_steps)

#         return out
    

class TemporalAttention(nn.Module):
    def __init__(self, time_steps: int):
        super().__init__()
        self.query = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)
        self.key = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)
        self.value = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)

        self.out_conv = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0 / (time_steps ** 0.5)))

    def forward(self, x):
        B, temp_out_channels, spat_channels, time_steps = x.shape

        x_reshaped = x.view(B * temp_out_channels, time_steps, spat_channels)
        query = self.query(x_reshaped)
        key = self.key(x_reshaped)
        value = self.value(x_reshaped)

        attn_scores = torch.einsum("btc,bts->bcs", query, key) * self.scale  
        attn_weights = torch.softmax(attn_scores, dim=-1)


        out = torch.einsum("bcs,bts->btc", attn_weights, value)
   
        out = out.reshape(B * temp_out_channels, time_steps,spat_channels)
        out = self.out_conv(out)
        out = out + x_reshaped 
        out = out.view(B, temp_out_channels, spat_channels, time_steps)
        return out



class ShallowAttentionNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.8, num_kernels=1, kernel_size=25, pool_size=200):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.mini_pool = nn.AvgPool2d((1,2))
        mini_steps = ((n_times - 2)//2)+1
        self.temporal_attention = TemporalAttention(mini_steps)
        self.spatial = nn.Conv2d(num_kernels, 22, (n_chans, 1))


        self.batch_norm = nn.BatchNorm2d(22) 

        self.pool = nn.AvgPool2d((1, pool_size))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, input):
        x = torch.unsqueeze(input, dim=1)  # Add channel dimension [B, 1, C, T]

        x = self.mini_pool(x)
        x = self.temporal_attention(x)  
        x = self.spatial(x)

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x