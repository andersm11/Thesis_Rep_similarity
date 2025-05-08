from CKA_functions import compute_all_model_kernels,compute_all_model_CKA,load_model_metadata,plot_cka_heatmaps
from CKA_functions import load_dataset,fix_dataset_shape,compute_cross_model_cka,display_cka_matrix,compute_multi_model_kernels,display_differences_matrix_og
from CKA_functions import compose_heat_matrix,compose_heat_matrix_shared,compose_heat_matrix_acc
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

compose_heat_matrix_acc("../cka_results","cka_heatmaps","../models","cka heatmap_fc_temporal")


