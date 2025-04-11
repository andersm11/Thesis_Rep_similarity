import sys
import importlib
import subprocess
import math
import torch
import numpy
import os
from CKA_functions import compute_all_model_kernels
from CKA_functions import compute_all_model_CKA_lowmem


model_direc = "models"
activation_direc = "activations"
kernel_direc = "kernels"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

X = torch.load('FACED_dataset/emotion_test_set.pt')
X = [sublist[0] for sublist in X]  # Extract the first tensor in each tuple
X = torch.stack(X).to(device) # Stack the tensors into a single tensor



layer_names=["sgconv","spatial","spatial_att"]
batch_size = 288
n_batches = 8
print("computing kernels",flush=True)
compute_all_model_kernels(model_direc,
                            activation_direc,
                            kernel_direc,X,
                            layer_names=layer_names,
                            batch_size=batch_size,
                            n_batches=n_batches,
                          device=device)
print("computing all model cka",flush=True)
compute_all_model_CKA_lowmem(kernel_direc,"cka_results")
