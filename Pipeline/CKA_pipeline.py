from CKA_functions import compute_multi_model_kernels,compute_cka_changes,display_differences_matrix
from CKA_functions import load_dataset,fix_dataset_shape,compute_cross_model_cka,display_cka_matrix
import numpy as np
import os
import torch
model_direc = "models"
activation_direc = "activations"
kernel_direc = "kernels"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
X = fix_dataset_shape(load_dataset("test_set.pkl","Datasets/")).to(device)
layer_names=["temporal","spatial_attention","pool","spatial","lstm"]
batch_size = 64
n_batches = 12
model_layer_names,model_names = compute_multi_model_kernels(model_direc,
                            activation_direc,
                            kernel_direc,X,
                            layer_names=layer_names,
                            batch_size=batch_size,
                            n_batches=n_batches)
cka_results = compute_cross_model_cka("kernels/")
# cka_results = np.array([[0.06508174 0.09691181 0.35968925]
#  [0.0248908  0.03481389 0.34109004]
#  [0.01694607 0.02389902 0.35218104]])
print("final:_", cka_results)
os.makedirs("ckaResults", exist_ok=True)
np.save("ckaResults/cka_results.npy", cka_results) 
np.savetxt("ckaResults/cka_results.csv", cka_results, delimiter=",")
display_cka_matrix(cka_results,model_layer_names[1],model_layer_names[0],model_names[1],model_names[0])
cka_differences =compute_cka_changes(cka_results)
print("differences:",cka_differences)
#display_differences_matrix(cka_differences,model_layer_names[0],model_layer_names[1],model_names[0],model_names[1])

