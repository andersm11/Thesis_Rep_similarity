from CKA_functions import compute_multi_model_kernels, load_dataset,fix_dataset_shape,compute_cross_model_cka,display_cka_matrix
import numpy as np
import os
model_direc = "models"
activation_direc = "activations"
kernel_direc = "kernels"
X = fix_dataset_shape(load_dataset("test_set.pkl","Datasets/"))
layer_names=["temporal","spatial_attention","pool","spatial"]
batch_size = 128
n_batches = 8
model_layer_names,model_names = compute_multi_model_kernels(model_direc,
                            activation_direc,
                            kernel_direc,X,
                            layer_names=layer_names,
                            batch_size=batch_size,
                            n_batches=n_batches)
cka_results = compute_cross_model_cka("kernels/")
print("final:_", cka_results)
os.makedirs("ckaResults", exist_ok=True)
np.save("ckaResults/cka_results.npy", cka_results)  # Save as NumPy .npy file
np.savetxt("ckaResults/cka_results.csv", cka_results, delimiter=",")
display_cka_matrix(cka_results,model_layer_names[0],model_layer_names[1],model_names[0],model_names[1])

