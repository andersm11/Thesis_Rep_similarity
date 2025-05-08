from CKA_functions import compute_all_model_kernels,compute_all_model_CKA,load_model_metadata,plot_cka_heatmaps
from CKA_functions import load_dataset,fix_dataset_shape,compute_cross_model_cka,display_cka_matrix,compute_multi_model_kernels,display_differences_matrix_og
from CKA_functions import compose_heat_matrix,compose_heat_matrix_shared
from torch.utils.data import DataLoader
from performance_functions import get_labels, compute_accuracy
import numpy as np
import os
import torch
model_direc = "../models"
activation_direc = "../activations"
kernel_direc = "../kernels"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
#X = fix_dataset_shape(load_dataset("test_set.pkl","../Datasets/"))
# X = torch.load("../Datasets/emotion_test_set.pt",map_location=device)
# X = [sublist[0] for sublist in X]  # Extract the first tensor in each tuple
# X = torch.stack(X)  # Stack the tensors into a single tensor

# print("data shape:", X.shape)

layer_names=["sgconv","spatial","spatial_att"]
batch_size = 288
n_batches = 8
# compute_all_model_kernels(model_direc,
#                             activation_direc,
#                             kernel_direc,X,
#                             layer_names=layer_names,
#                             batch_size=batch_size,
#                             n_batches=n_batches)
# compute_all_model_CKA(kernel_direc,"../cka_results")
# plot_cka_heatmaps("../cka_results","../kernels")
# X = torch.load('../Datasets/emotion_test_set.pt')
test_loader = DataLoader(layer_names, batch_size=16)
compose_heat_matrix_shared("../cka_results","cka_heatmaps","../Shared_Keys","cka heatmap_shared_fc_temporal")


