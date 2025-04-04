from CKA_functions import compute_all_model_kernels,compute_all_model_CKA,load_model_metadata,plot_cka_heatmaps
from CKA_functions import load_dataset,fix_dataset_shape,compute_cross_model_cka,display_cka_matrix,compute_multi_model_kernels,display_differences_matrix_og
from CKA_functions import compose_heat_matrix
import numpy as np
import os
import torch
model_direc = "../models"
activation_direc = "../activations"
kernel_direc = "../kernels"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
X = fix_dataset_shape(load_dataset("test_set.pkl","../Datasets/"))
#X = torch.stack(X)




layer_names=["temporal","sgconv","spatial","pool","fc"]
batch_size = 128
n_batches = 8
# model_layer_names, model_names = compute_multi_model_kernels(model_direc,
#                             activation_direc,
#                             kernel_direc,X,
#                             layer_names=layer_names,
#                             batch_size=batch_size,
#                             n_batches=n_batches)
# compute_all_model_kernels(model_direc,
#                             activation_direc,
#                             kernel_direc,X,
#                             layer_names=layer_names,
#                             batch_size=batch_size,
#                             n_batches=n_batches)
plot_cka_heatmaps("../cka_results","../kernels")


