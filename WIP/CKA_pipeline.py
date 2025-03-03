from CKA_functions import compute_multi_model_kernels, load_dataset,fix_dataset_shape,compute_cross_model_cka

model_direc = "models"
activation_direc = "activations"
kernel_direc = "kernels"
X = fix_dataset_shape(load_dataset("test_set.pkl","Datasets/"))
layer_names=["temporal","spatial_attention","pool","spatial"]
batch_size = 64
n_batches = 2
compute_multi_model_kernels(model_direc,
                            activation_direc,
                            kernel_direc,X,
                            layer_names=layer_names,
                            batch_size=batch_size,
                            n_batches=n_batches)
bla = compute_cross_model_cka("kernels/")
print("final:_", bla)

