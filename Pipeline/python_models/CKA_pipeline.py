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




layer_names=["sgconv","spatial","pool","fc"]
batch_size = 128
n_batches = 4
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
# compute_all_model_CKA(kernel_direc,"cka_results")
#cka_results = compute_all_model_CKA(kernel_direc,"cka_temp_results")
# cka_results = np.array([[0.89855301 ,0.22526194 ,0.11029866, 0.08708372],
#  [0.49409801, 0.4049519 , 0.29927254, 0.242651  ],
#  [0.28096467 ,0.52176714 ,0.56983662, 0.47695962],
#  [0.14419821, 0.37224442 ,0.56296223, 0.70330274]])
# print("final:_", cka_results)
# os.makedirs("ckaResults", exist_ok=True)
# np.save("ckaResults/cka_results.npy", cka_results) 
# np.savetxt("ckaResults/cka_results.csv", cka_results, delimiter=",")
#display_differences_matrix_og(cka_results,model_layer_names[0],model_layer_names[1],model_names[0],model_names[1])
# cka_differences =compute_cka_changes(cka_results)
# print("differences:",cka_differences)
# display_differences_matrix(cka_differences,model_layer_names[0],model_layer_names[1],model_names[0],model_names[1])
plot_cka_heatmaps("cka_results","../kernels")


