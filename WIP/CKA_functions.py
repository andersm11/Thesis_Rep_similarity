import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from shallow_fbcsp import ShallowFBCSPNet
from braindecode.models import EEGConformer
import Attention_models
import importlib
import pandas as pd
from collections import OrderedDict
import math
import pickle
import os
from typing import Type,Optional
from itertools import product


# CKA math

def linear_kernel(X):
    """Computes the linear kernel matrix for X."""
    return torch.matmul(X,X.T)  # Dot product

def centering_matrix(K):
    """Apply centering to the kernel matrix."""
    n = K.shape[0]
    H = torch.eye(n) - (1.0 / n) * torch.ones((n, n), device=K.device)
    return H @ K @ H  # Centered kernel matrix

def compute_hsic(K_x, K_y):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC).
    
    Parameters:
    - X: (n_samples, n_features_X) numpy array
    - Y: (n_samples, n_features_Y) numpy array
    - kernel_X: function to compute the kernel matrix for X
    - kernel_Y: function to compute the kernel matrix for Y
    
    Returns:
    - HSIC value (float)
    """
    K_x_centered = centering_matrix(K_x)
    K_y_centered = centering_matrix(K_y)
    hsic_value = np.trace(K_x_centered @ K_y_centered) / ((K_x.shape[0] - 1) ** 2)
    return hsic_value
  
def CKA(K_x,K_y):
  """
  compute CKA between two X,Y activations
  
  Parameters:
  - X: (n_samples, x_features)
  - Y: (n_samples, y_features)
  - kernel_X: kernel for X
  - kernel_Y: kernel for Y  
  """
  HSIC_KL = compute_hsic(K_x,K_y) 
  HSIC_KK = compute_hsic(K_x,K_x)
  HSIC_LL = compute_hsic(K_y,K_y)
  numerator = HSIC_KL
  denominator = math.sqrt(HSIC_KK * HSIC_LL)
  return(numerator/denominator).item()



# Load Model
def load_model(
    model_name: str, 
    path: str = "",
    load_state: bool = False,
    model_class: Type[torch.nn.Module] = None,
    *args,
    **kargs
):
    if not path.endswith('/'):
        path = path + '/'
    full_model_path = os.path.join(path, model_name)

    if load_state:
        if model_class is None:
            raise ValueError("Load_state is set, but model_class is None.\nAppropriate model must be given")
        else:
            model = model_class(*args, **kargs)
            model.load_state_dict(torch.load(full_model_path, map_location=torch.device('cpu')))
            return model
    else:
        # Additional logic if you don't need to load the model state (this part is assumed)
        model = torch.load(full_model_path,weights_only = False,map_location=torch.device('cpu'))
        return model
    
    
# Load Dataset
def load_dataset(file_name: str, path:str = ""):
    if not path.endswith('/'):
        path = path + '/'
    full_file_path = os.path.join(path, file_name)
    with open(full_file_path,'rb') as f:
        test_set = pickle.load(f)
        return test_set
    

# Fix dataset shape
def fix_dataset_shape(data):
    x = torch.stack([torch.from_numpy(data[i][0]) for i in range(len(data))])
    return x


#Extract activation for a single model
#Extract activation for a single model
def extract_model_activations(model: torch.nn.Module, input_tensor: torch.Tensor, output_dir: str, layer_names: list[str], batch_size: int = 128):
    found_layer_names = []
    # Check if the output directory exists and is not empty
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Output directory {output_dir} is not empty. Returning found layer names.")
        for name, layer in model.named_modules():
            if name in layer_names:
                found_layer_names.append(name)
        return found_layer_names  # Return the found layer names if the directory isn't empty

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    activations = OrderedDict()
    

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks for specific layers
    for name, layer in model.named_modules():
        if name in layer_names:
            layer.register_forward_hook(get_activation(name))
            found_layer_names.append(name)

    model.eval()

    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[i:i + batch_size]  # Select current batch
            _ = model(batch)  # Forward pass through the model

            # Save activations after each batch
            for name, activation in activations.items():
                batch_idx = i // batch_size + 1  # This determines the batch number
                print(f"saving: {name}_batch_{batch_idx}.pt")
                torch.save(activation, os.path.join(output_dir, f"{name}_batch_{batch_idx}.pt"))
            
            # Clear activations list after saving
            activations.clear()
            torch.cuda.empty_cache()
            
    return found_layer_names

# compute kernel single model
def compute_kernel_full_lowmem(layer, total_nr_batches:int, batch_size:int, total_samples: int, load_dir:str, n_batches: int =1, use_cuda=False):
    """Computes the full kernel matrix in batches efficiently using matrix multiplication."""
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    full_kernel = torch.zeros((total_samples, total_samples), dtype=torch.float32, device=device)
    
    for batch_idx in range(1, math.ceil(math.ceil(total_nr_batches) / n_batches) + 1):
        print(f"\rLoading batches {batch_idx * n_batches - (n_batches - 1)}-{batch_idx * n_batches} for {layer} in {load_dir}")

        batch_activations_list = []
        start_idx_col = (batch_idx - 1) * batch_size * n_batches
        end_idx_col = min(start_idx_col + batch_size * n_batches, total_samples)  

        # Load multiple batches at once
        for sub_batch in range(n_batches):
            batch_file_idx = (batch_idx - 1) * n_batches + sub_batch + 1
            if batch_file_idx > math.ceil(total_nr_batches):
                break
            batch_activations = torch.load(
                f"{load_dir}/{layer}_batch_{batch_file_idx}.pt"
            ).to(device)
            batch_activations_list.append(batch_activations.reshape(batch_activations.shape[0], -1))

        if not batch_activations_list:
            continue

        batch_activations = torch.cat(batch_activations_list, dim=0)  # Merge batches

        for batch_idx2 in range(1, math.ceil(math.ceil(total_nr_batches) / n_batches) + 2):
            batch_activations_transpose_list = []
            start_idx_row = (batch_idx2 - 1) * batch_size * n_batches
            end_idx_row = min(start_idx_row + batch_size * n_batches, total_samples)

            # Load multiple batches at once
            for sub_batch2 in range(n_batches):
                batch_file_idx2 = (batch_idx2 - 1) * n_batches + sub_batch2 + 1
                if batch_file_idx2 > math.ceil(total_nr_batches):
                    break
                batch_activations_transpose = torch.load(
                    f"{load_dir}/{layer}_batch_{batch_file_idx2}.pt"
                ).to(device)
                batch_activations_transpose_list.append(batch_activations_transpose.reshape(batch_activations_transpose.shape[0], -1))

            if not batch_activations_transpose_list:
                continue

            batch_activations_transpose = torch.cat(batch_activations_transpose_list, dim=0)

            kernel_block = batch_activations @ batch_activations_transpose.T
            full_kernel[start_idx_col:end_idx_col, start_idx_row:end_idx_row] = kernel_block
            full_kernel[start_idx_row:end_idx_row, start_idx_col:end_idx_col] = kernel_block.T  # Use symmetry

    return full_kernel.cpu()


def compute_full_kernels(layer_names: list[str], total_nr_batches: int, batch_size:int, total_samples: int, load_dir:str,  save_dir:str, n_batches:int = 1, use_cuda:bool=False):
    """Computes the kernels for model and save them to the given directory."""
    
    # Ensure the save directory exists
    #if not os.path.exists(save_dir):
        #os.makedirs(save_dir)
    model_kernels = {}
    # Compute and save kernels for model 1
    for layer in layer_names:
        # Compute the kernel
        kernel = compute_kernel_full_lowmem(layer, total_nr_batches, batch_size, total_samples, load_dir, n_batches, use_cuda)
        #print(kernel)
        print("got layer: ", layer)
        # print("i got layer:::   ", layer)
        # if (kernel[layer] == 0).any():
        #     print("The tensor contains at least one zero value.")
        model_kernels[layer] = kernel

    #_, kernel_filename = save_dir.rsplit('/',1)
    #kernel_path = os.path.join(save_dir, kernel_filename)
    save_path, _ = save_dir.rsplit('.',1)
    torch.save(model_kernels, save_path)
    print(f"Saved kernels for model at {save_dir}")


#compute kernels for all models in directory
def compute_multi_model_kernels(
    models_directory:str, 
    activations_root_directory:str,
    kernels_directory:str, 
    input_data:torch.Tensor, 
    layer_names:list[str], 
    use_state:bool = False, 
    batch_size:int=128,
    n_batches:int = 1,
    use_cuda:bool = False
):
    model_files = os.listdir(models_directory)
    total_samples = input_data.shape[0]
    total_nr_batches = math.ceil(total_samples/batch_size)
    for model_file in model_files:
        if not model_file.endswith('state.pth'):
            model_name, loss, seed = model_file.rsplit('_', 2)
            # model_path = os.path.join(models_directory, model_file)
            model = load_model(model_file,models_directory)
            model.eval() 
            found_names = extract_model_activations(model, input_data, activations_root_directory+f"/{model_name}/{loss}_{seed}",layer_names, batch_size=batch_size)
            compute_full_kernels(
                found_names,     
                total_nr_batches,
                batch_size,total_samples,
                activations_root_directory+f"/{model_name}/{loss}_{seed}",
                kernels_directory+f"/{model_name}/{loss}_{seed}",
                n_batches,
                use_cuda
            )
            
            

def compute_cross_model_cka(root_dir: str):
    """
    Computes the average CKA similarity matrix between layers of two model types.

    Args:
        root_dir (str): Path to the directory containing two subdirectories (one per model type).

    Returns:
        np.array: A 2D NumPy array representing the CKA similarity matrix.
    """
    model_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    if len(model_dirs) != 2:
        raise ValueError("Expected exactly two model type directories in the root directory.")

    # Load kernels for both model types into lists
    model_type1_kernels = []  # For the first model type
    model_type2_kernels = []  # For the second model type

    for i, model_dir in enumerate(model_dirs):
        for filename in os.listdir(model_dir):
            model_path = os.path.join(model_dir, filename)
            kernel = torch.load(model_path)  # Load kernel dictionary

            # Append the kernel for each model type into the respective list
            if i == 0:
                model_type1_kernels.append(kernel)  # For the first model type
            else:
                model_type2_kernels.append(kernel)  # For the second model type
    print("llllllen:",len(model_type1_kernels))
    print("llllllen2:",len(model_type1_kernels))
    
        # Initialize CKA matrix (assuming all models have the same number of kernels for simplicity)
    # Initialize the results dictionary and matrix
    cka_results = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))

    # Create a mapping from layer names to indices
    layer1_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type1_kernels[0].keys())}
    layer2_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type2_kernels[0].keys())}

    # Compute CKA between kernels from both model types
    for i, kernel_A in enumerate(model_type1_kernels):
        cka_inner = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))  # Initialize inner CKA matrix for each kernel_A
        
        # For each kernel in model_type2
        for j, kernel_B in enumerate(model_type2_kernels):
            # Compute CKA for each layer pair between kernel_A and kernel_B
            for layer1, K_x in kernel_A.items():
                for layer2, K_y in kernel_B.items():
                    cka_value = CKA(K_x, K_y)  # Compute CKA between this pair of kernels
                    # Convert layer names to indices
                    idx1 = layer1_to_idx[layer1]
                    idx2 = layer2_to_idx[layer2]
                    
                    # Accumulate the CKA value for the current layer pair
                    cka_inner[idx1, idx2] += cka_value
                    print(f"CKA({layer1}, {layer2}): {cka_value}")
            
        # Average the CKA values for each layer pair after looping through all kernels in model_type2
        cka_inner /= len(model_type2_kernels)
        
        # Add the averaged CKA values to the overall results
        cka_results += cka_inner
        
        print(f"Avg CKA result for kernel {i}: {cka_inner}")

    # After finishing with all kernel_A, average the CKA values across model_type1 kernels
    cka_results /= len(model_type1_kernels)

    return cka_results  # Return the CKA similarity matrix

            

            
