import sys
import importlib
import subprocess
import math
import torch
import numpy
import os
from CKA_functions import compute_all_model_kernels_indexed
from CKA_functions import compute_all_model_CKA_lowmem
import shutil
model_direc = "cka_models"
activation_direc = "activations2"
kernel_direc = "kernels2"
keyfile_folder = "Shared_Keys"
cka_folder = "cka_results2"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# Load and prepare input data
X = torch.load('FACED_dataset/emotion_test_set.pt')
X = [sublist[0] for sublist in X]  # Extract the first tensor in each tuple
X = torch.stack(X).to(device)     # Stack and move to device

layer_names = ["spatial","sgconv","spatial_att"]
batch_size = 288
n_batches = 6

# Iterate over all CSV key files in the keyfile directory
for file in os.listdir(keyfile_folder):
    if file.endswith(".csv") and file.startswith("Shared_Keys_"):
        keyfile_path = os.path.join(keyfile_folder, "Shared_Keys_Spatial.csv")
        print(f"Computing kernels for: {keyfile_path}", flush=True)

        compute_all_model_kernels_indexed(
            model_direc,
            activation_direc,
            kernel_direc,
            X,
            layer_names=layer_names,
            batch_size=batch_size,
            n_batches=n_batches,
            device=device,
            keyfile_path=keyfile_path
        )
    print("computing all model cka",flush=True)
    compute_all_model_CKA_lowmem(kernel_direc,cka_folder)
    for root, dirs, files in os.walk(kernel_direc):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
