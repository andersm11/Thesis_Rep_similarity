import json
import os
import itertools
import numpy as np
import logging
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from collections import OrderedDict
import math
import pickle
import os
from typing import Type,Optional
import torch.nn.functional as F
import os

# CKA math

def linear_kernel(X,Xt):
    """Computes the linear kernel matrix for X."""
    return X @ Xt 

def cosine_kernel(X, Xt=None):
    if Xt is None:
        Xt = X
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Xt_norm = Xt / (Xt.norm(dim=1, keepdim=True) + 1e-8)
    return torch.matmul(X_norm, Xt_norm.T) 

def polynomial_kernel(X, Xt=None, degree=2, c=0.01):
    """Computes the polynomial kernel."""
    if Xt is None:
        Xt = X
    return (torch.matmul(X, Xt.T) + c) ** degree


def rbf_kernel(X,Xt, sigma=None):
    """Computes the RBF (Gaussian) kernel matrix."""
    pairwise_sq_dists = torch.cdist(X, Xt, p=2) ** 2 
    if sigma is None:
        sigma = torch.median(pairwise_sq_dists).sqrt()
    return torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))

def polynomial_kernel(X, Y, degree=2, c=1):
    """Computes the polynomial kernel matrix."""
    return (X @ Y.T + c) ** degree

def normalize_kernel(K):
    """Normalize the kernel matrix by its trace."""
    trace_K = torch.trace(K)
    return K / trace_K 

def centering_matrix(K):
    """Apply centering to the kernel matrix."""
    n = K.shape[0]
    H = torch.eye(n,device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
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
    hsic_value = torch.trace(K_x_centered @ K_y_centered) / ((K_x.shape[0] - 1) ** 2)
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


def extract_model_activations(model: torch.nn.Module, input_tensor: torch.Tensor, output_dir: str, layer_names: list[str], batch_size: int = 128,device='cpu'):
    found_layer_names = []
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Output directory {output_dir} is not empty, assuming kernel already computed. Returning found layer names.")
        for name, layer in model.named_modules():
            if name in layer_names:
                found_layer_names.append(name)
        return found_layer_names  # Return the found layer names if the directory isn't empty
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    activations = OrderedDict()
    

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):  
                output = output[0]  # Extract only the first element (LSTM output)
            activations[name] = output.detach()
        return hook
    for name, layer in model.named_modules():
        if name in layer_names:
            layer.register_forward_hook(get_activation(name))
            found_layer_names.append(name)

    model.eval()
    try:
        from SGCN_FACED_norm import ShallowSGCNNet
    except ImportError as e1:
        try:
            from SGCN import ShallowSGCNNet
        except ImportError as e2:
            print("Failed to import from both modules:")
            print("SGCN_FACED:", e1)
            print("SGCN:", e2) 
    if isinstance(model, ShallowSGCNNet) and (ShallowSGCNNet.__module__ == 'SGCN' or ShallowSGCNNet.__module__ == 'RGNN'):
        adj_m,pos = adjacency_matrix_motion()
        adj_dis_m, dm = adjacency_matrix_distance_motion(pos,delta=10)
        threshold = 0  
        source_nodes = []
        target_nodes = []

        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):  
                if dm[i, j] >= threshold:  
                    source_nodes.append(i)  
                    target_nodes.append(j) 
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    elif isinstance(model, ShallowSGCNNet) and ShallowSGCNNet.__module__ == 'SGCN_FACED_norm':
        adj_m,pos = adjacency_matrix_FACED()
        adj_dis_m, dm = adjacency_matrix_distance_FACED(pos,delta=5)
        threshold = 0 
        source_nodes = []
        target_nodes = []
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]): 
                if dm[i, j] >= threshold: 
                    source_nodes.append(i)  
                    target_nodes.append(j)  
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[i:i + batch_size] 
            if isinstance(model,ShallowSGCNNet):
                _ = model(batch,edge_index.to(device))
            else:
                _ = model(batch) 
            for name, activation in activations.items():
                batch_idx = i // batch_size + 1  
                print(f"saving: {name}_batch_{batch_idx}.pt")
                torch.save(activation, os.path.join(output_dir, f"{name}_batch_{batch_idx}.pt"))
            activations.clear()
            torch.cuda.empty_cache()
            
    return found_layer_names

def extract_model_activations_indexed(
    model: torch.nn.Module, 
    input_tensor: torch.Tensor, 
    output_dir: str, 
    layer_names: list[str], 
    batch_size: int = 128,
    device='cpu',
    index_csv_path: str = 'Shared_Keys_Shallow_and_RNN.csv'
):
    # Load index CSV
    df = pd.read_csv(index_csv_path)
    valid_indices = df['index'].tolist()

    # Filter input_tensor
    selected_tensor = input_tensor[valid_indices]

    found_layer_names = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    activations = OrderedDict()

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        if name in layer_names:
            layer.register_forward_hook(get_activation(name))
            found_layer_names.append(name)

    model.eval()

    try:
        from SGCN_FACED_norm import ShallowSGCNNet
    except ImportError as e1:
        try:
            from SGCN_norm import ShallowSGCNNet
        except ImportError as e2:
            print("Failed to import from both modules:")
            print("SGCN_FACED:", e1)
            print("SGCN:", e2)

    if isinstance(model, ShallowSGCNNet) and (ShallowSGCNNet.__module__ == 'SGCN' or ShallowSGCNNet.__module__ == 'RGNN'):
        adj_m, pos = adjacency_matrix_motion()
        adj_dis_m, dm = adjacency_matrix_distance_motion(pos, delta=10)
        threshold = 0
        source_nodes, target_nodes = [], []
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                if dm[i, j] >= threshold:
                    source_nodes.append(i)
                    target_nodes.append(j)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    elif isinstance(model, ShallowSGCNNet) and ShallowSGCNNet.__module__ == 'SGCN_FACED_norm':
        adj_m, pos = adjacency_matrix_FACED()
        adj_dis_m, dm = adjacency_matrix_distance_FACED(pos, delta=5)
        threshold = 0
        source_nodes, target_nodes = [], []
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                if dm[i, j] >= threshold:
                    source_nodes.append(i)
                    target_nodes.append(j)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    batch_counter = 0  
    total_samples = len(valid_indices)
    with torch.no_grad():
        for i in range(0, selected_tensor.shape[0], batch_size):
            batch = selected_tensor[i:i + batch_size].to(device)
            if isinstance(model, ShallowSGCNNet):
                _ = model(batch, edge_index.to(device))
            else:
                _ = model(batch)
            for name, activation in activations.items():
                batch_idx = i // batch_size + 1
                print(f"saving: {name}_batch_{batch_idx}.pt")
                torch.save(activation, os.path.join(output_dir, f"{name}_batch_{batch_idx}.pt"))
            activations.clear()
            torch.cuda.empty_cache()
            batch_counter+=1
    return found_layer_names,batch_counter,total_samples

# compute kernel single model
def compute_kernel_full_lowmem(layer, total_nr_batches:int, batch_size:int, total_samples: int, load_dir:str, n_batches: int =1, device='cpu'):
    """Computes the full kernel matrix in batches efficiently using matrix multiplication."""
    

    full_kernel = torch.zeros((total_samples, total_samples), dtype=torch.float32, device=device)
    
    for batch_idx in range(1, math.ceil(math.ceil(total_nr_batches) / n_batches) + 1):
        print(f"\rLoading batches {batch_idx * n_batches - (n_batches - 1)}-{batch_idx * n_batches} for {layer} in {load_dir}")

        batch_activations_list = []
        start_idx_col = (batch_idx - 1) * batch_size * n_batches
        end_idx_col = min(start_idx_col + batch_size * n_batches, total_samples)  

        for sub_batch in range(n_batches):
            batch_file_idx = (batch_idx - 1) * n_batches + sub_batch + 1
            if batch_file_idx > math.ceil(total_nr_batches):
                break
            batch_activations = torch.load(
                f"{load_dir}/{layer}_batch_{batch_file_idx}.pt",weights_only=True).to(device)
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
                    f"{load_dir}/{layer}_batch_{batch_file_idx2}.pt", weights_only= False
                ).to(device)
                batch_activations_transpose_list.append(batch_activations_transpose.reshape(batch_activations_transpose.shape[0], -1))

            if not batch_activations_transpose_list:
                continue
            batch_activations.to(device)
            batch_activations_transpose.to(device)
            batch_activations_transpose = torch.cat(batch_activations_transpose_list, dim=0)
            kernel_block = linear_kernel(batch_activations,batch_activations_transpose.T)
            #kernel_block = rbf_kernel(batch_activations,batch_activations_transpose,sigma=None)

            full_kernel[start_idx_col:end_idx_col, start_idx_row:end_idx_row] = kernel_block.to('cpu')
            full_kernel[start_idx_row:end_idx_row, start_idx_col:end_idx_col] = kernel_block.T  # Use symmetry
    return full_kernel.cpu()


import os
import torch

def compute_full_kernels(layer_names: list[str], total_nr_batches: int, batch_size: int, total_samples: int, load_dir: str, id:str,save_dir: str, n_batches: int = 1, device = 'cpu'):
    """Computes the kernels for model and saves them to the given directory if not already saved.
    Deletes all elements in load_dir after computation and adds an empty 'done' file."""
    
    save = save_dir+f"/{id}"
    if os.path.exists(save):
        print(f"Kernel file already exists at {save}. Skipping computation.")
        return
    
    model_kernels = {}
    
    for layer in layer_names:
        print("Got layer:", layer)
        kernel = compute_kernel_full_lowmem(layer, total_nr_batches, batch_size, total_samples, load_dir, n_batches, device)
        print("kernelsize::: ", kernel.shape)
        model_kernels[layer] = kernel
    
    torch.save(model_kernels, save)
    print(f"Saved kernels for model at {save}")
    
    for file_name in os.listdir(load_dir):
        file_path = os.path.join(load_dir, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
    print(f"Cleared all elements in {load_dir}")
    
    done_file_path = os.path.join(load_dir, 'done')
    with open(done_file_path, 'w') as f:
        pass  # Just create an empty file
    print(f"Added empty 'done' file to {load_dir}")


def compute_multi_model_kernels(
    models_directory: str,
    activations_root_directory: str,
    kernels_directory: str,
    input_data: torch.Tensor,
    layer_names: list[str],
    use_state: bool = False,
    batch_size: int = 128,
    n_batches: int = 1,
    device: Optional[torch.device] = 'cpu',
):
    """Computes kernels for multiple models and saves them in the given directory."""
    model_files = os.listdir(models_directory)
    total_samples = input_data.shape[0]
    total_nr_batches = math.ceil(total_samples / batch_size)
    final_layer_names=[]
    final_model_names=[]
    for model_file in model_files:
        if not model_file.endswith('state.pth') and not model_file.startswith('.'):
            print("Model file:", model_file)
            model_name, loss, seed = model_file.rsplit('_', 2)
            model = load_model(model_file, models_directory,load_state=False)
            model.to(device)
            model.eval()
            input_data = input_data.to(device)
            print(f"Input data is on device: {input_data.device}")
            print(f"model is on device: {next(model.parameters()).device}")
            activation_dir = os.path.join(activations_root_directory, model_name, f"{loss}_{seed}")
            try:
                found_names = extract_model_activations(model, input_data, activation_dir, layer_names, batch_size=batch_size,device=device)
            except Exception as e:
                print(f"Error extracting activations: {e}. Resampling input data to 1000 samples and retrying.")
                downsampled_data = F.interpolate(input_data, size=(1000,), mode='linear', align_corners=False)
                found_names = extract_model_activations(model, downsampled_data, activation_dir, layer_names, batch_size=batch_size,device=device)
            if not (model_name in final_model_names):
                final_model_names.append(model_name)
                final_layer_names.append(found_names)
            kernel_dir = os.path.join(kernels_directory, model_name)
            os.makedirs(kernel_dir, exist_ok=True)  # Ensure kernel directory exists
            compute_full_kernels(
                found_names,
                total_nr_batches,
                batch_size,
                total_samples,
                activation_dir,
                f"{loss}_{seed}",
                kernel_dir,
                n_batches,
                device
            )
    return final_layer_names, final_model_names

def compute_multi_model_kernels_indexed(
    models_directory: str,
    activations_root_directory: str,
    kernels_directory: str,
    input_data: torch.Tensor,
    layer_names: list[str],
    use_state: bool = False,
    batch_size: int = 128,
    n_batches: int = 1,
    device: Optional[torch.device] = 'cpu',
    keyfile_path: str = 'Shared_Keys_Shallow_and_RNN.csv'
):
    """Computes kernels for multiple models and saves them in the given directory."""
    model_files = os.listdir(models_directory)
    total_samples = input_data.shape[0]
    total_nr_batches = math.ceil(total_samples / batch_size)
    final_layer_names=[]
    final_model_names=[]
    for model_file in model_files:
        if not model_file.endswith('state.pth'):
            print("splitting model file:", model_file)
            model_name, loss, seed = model_file.rsplit('_', 2)
            model = load_model(model_file, models_directory,load_state=False)
            model.to(device)
            model.eval()
            
            activation_dir = os.path.join(activations_root_directory, model_name, f"{loss}_{seed}")
            try:
                found_names,nr_batches,total_samples = extract_model_activations_indexed(model, input_data, activation_dir, layer_names, batch_size=batch_size,device=device,index_csv_path=keyfile_path)
            except Exception as e:
                print(f"Error extracting activations: {e}. Resampling input data to 1000 samples and retrying.")
                downsampled_data = F.interpolate(input_data, size=(1000,), mode='linear', align_corners=False)
                found_names,nr_batches,total_samples = extract_model_activations_indexed(model, downsampled_data, activation_dir, layer_names, batch_size=batch_size,device=device,index_csv_path=keyfile_path)
            if not (model_name in final_model_names):
                final_model_names.append(model_name)
                final_layer_names.append(found_names)
            kernel_dir = os.path.join(kernels_directory, model_name)
            os.makedirs(kernel_dir, exist_ok=True)  # Ensure kernel directory exists
            compute_full_kernels(
                found_names,
                nr_batches,
                batch_size,
                total_samples,
                activation_dir,
                f"{loss}_{seed}",
                kernel_dir,
                n_batches,
                device
            )
    return final_layer_names, final_model_names
            
            
def compute_all_model_kernels(
    target_directory: str,
    activations_root_directory: str,
    kernels_directory: str,
    input_data: torch.Tensor,
    layer_names: list[str],
    use_state: bool = False,
    batch_size: int = 128,
    n_batches: int = 1,
    device: Optional[torch.device] = 'cpu',
):
    model_directories = os.listdir(target_directory)
    kwargs = {
        "activations_root_directory": activations_root_directory,
        "kernels_directory": kernels_directory,
        "input_data": input_data,
        "layer_names": layer_names,
        "use_state": use_state,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "device": device,
    }
    for direc in model_directories:
        model_path = os.path.join(target_directory,direc)
        layer_list,model_list = compute_multi_model_kernels(model_path, **kwargs)
        model_name = model_list[0]
        kernel_specific_path = os.path.join(kernels_directory,model_name)
        with open(os.path.join(kernel_specific_path, f"{direc}_list1.json"), "w") as f_layer:
            json.dump(layer_list, f_layer, indent=4)
        with open(os.path.join(kernel_specific_path, f"{direc}_list2.json"), "w") as f_model:
            json.dump(model_list, f_model, indent=4)

def compute_all_model_kernels_indexed(
    target_directory: str,
    activations_root_directory: str,
    kernels_directory: str,
    input_data: torch.Tensor,
    layer_names: list[str],
    use_state: bool = False,
    batch_size: int = 128,
    n_batches: int = 1,
    device: Optional[torch.device] = 'cpu',
    keyfile_path: str = 'Shared_Keys_Shallow_and_RNN.csv'
):
    # Map human-readable model names to folder names
    model_name_map = {
        "Attention": "ShallowAtt",
        "Shallow": "ShallowFBCSP",
        "LSTM": "ShallowLSTM",
        "RNN": "ShallowRNN",
        "SGCN": "ShallowSGCN"
    }

    # Parse keyfile to get model pair
    try:
        basename = os.path.basename(keyfile_path)
        model1_str, model2_str = basename.replace("Shared_Keys_", "").replace(".csv", "").split("_and_")
        target_folders = {model_name_map[model1_str], model_name_map[model2_str]}
    except Exception as e:
        raise ValueError(f"Failed to parse model names from keyfile `{keyfile_path}`: {e}")

    # Prepare common kwargs
    kwargs = {
        "activations_root_directory": activations_root_directory,
        "kernels_directory": kernels_directory,
        "input_data": input_data,
        "layer_names": layer_names,
        "use_state": use_state,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "device": device,
        "keyfile_path": keyfile_path
    }


    model_directories = os.listdir(target_directory)
    for direc in model_directories:
        if direc not in target_folders:
            continue  # skip unrelated models

        model_path = os.path.join(target_directory, direc)
        print("Processing model:", model_path)
        layer_list, model_list = compute_multi_model_kernels_indexed(model_path, **kwargs)

        if not model_list:
            continue  

        model_name = model_list[0]
        kernel_specific_path = os.path.join(kernels_directory, model_name)
        os.makedirs(kernel_specific_path, exist_ok=True)

        with open(os.path.join(kernel_specific_path, f"{direc}_list1.json"), "w") as f_layer:
            json.dump(layer_list, f_layer, indent=4)
        with open(os.path.join(kernel_specific_path, f"{direc}_list2.json"), "w") as f_model:
            json.dump(model_list, f_model, indent=4)


def compute_all_model_kernels_indexed_all(
    target_directory: str,
    activations_root_directory: str,
    kernels_directory: str,
    input_data: torch.Tensor,
    layer_names: list[str],
    use_state: bool = False,
    batch_size: int = 128,
    n_batches: int = 1,
    device: Optional[torch.device] = 'cpu',
    keyfile_path: str = 'Shared_Keys_Shallow_and_RNN.csv'
):

    model_name_map = {
        "Attention": "ShallowAtt",
        "Shallow": "ShallowFBCSP",
        "LSTM": "ShallowLSTM",
        "RNN": "ShallowRNN",
        "SGCN": "ShallowSGCN"
    }

    # Prepare commmon kwargs
    kwargs = {
        "activations_root_directory": activations_root_directory,
        "kernels_directory": kernels_directory,
        "input_data": input_data,
        "layer_names": layer_names,
        "use_state": use_state,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "device": device,
        "keyfile_path": keyfile_path
    }

    model_directories = os.listdir(target_directory)
    for direc in model_directories:


        model_path = os.path.join(target_directory, direc)
        print("Processing model:", model_path)
        layer_list, model_list = compute_multi_model_kernels_indexed(model_path, **kwargs)

        if not model_list:
            continue  # skip if model failed

        model_name = model_list[0]
        kernel_specific_path = os.path.join(kernels_directory, model_name)
        os.makedirs(kernel_specific_path, exist_ok=True)

        with open(os.path.join(kernel_specific_path, f"{direc}_list1.json"), "w") as f_layer:
            json.dump(layer_list, f_layer, indent=4)
        with open(os.path.join(kernel_specific_path, f"{direc}_list2.json"), "w") as f_model:
            json.dump(model_list, f_model, indent=4)

def compute_cross_model_cka(root_dir: str):
    """
    Computes the average CKA similarity matrix between layers of two model types.

    Args:
        root_dir (str): Path to the directory containing two subdirectories (one per model type).

    Returns:
        np.array: A 2D NumPy array representing the CKA similarity matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if len(model_dirs) != 2:
        raise ValueError("Expected exactly two model type directories in the root directory.")

    model_type1_kernels = []  
    model_type2_kernels = [] 
    model_name1 = ""
    model_name2 = ""

    for i, model_dir in enumerate(model_dirs):
        for filename in os.listdir(model_dir):
            model_path = os.path.join(model_dir, filename)
            kernel = torch.load(model_path, map_location=device,weights_only=False)  

            if i == 0:
                model_type1_kernels.append(kernel)  
                _, model_name1 = model_dir.rsplit('/', 1)
            else:
                _, model_name2 = model_dir.rsplit('/', 1)
                model_type2_kernels.append(kernel)  

    cka_results = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))

    layer1_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type1_kernels[0].keys())}
    layer2_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type2_kernels[0].keys())}

    for i, kernel_A in enumerate(model_type1_kernels):
        cka_inner = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))  # Initialize inner CKA matrix for each kernel_A
        
        for j, kernel_B in enumerate(model_type2_kernels):
            for layer1, K_x in kernel_A.items():
                for layer2, K_y in kernel_B.items():
                    K_x, K_y = K_x.to(device), K_y.to(device)
                    
                    cka_value = CKA(K_x, K_y)  
                    idx1 = layer1_to_idx[layer1]
                    idx2 = layer2_to_idx[layer2]
                    
                    cka_inner[idx1, idx2] += cka_value
                    print(f"CKA({model_name1}.{layer1}, {model_name2}.{layer2}): {cka_value}")
            
        cka_inner /= len(model_type2_kernels)
        cka_results += cka_inner
        
        
        print(f"Avg CKA result for kernel {i}: {cka_inner}")

    cka_results /= len(model_type1_kernels)
    return cka_results  # Return the CKA similarity matrix


def compute_all_model_CKA(root_dir: str, output_dir: str):
    """
    Computes CKA between models in different folders under the root directory.
    Each folder represents a model architecture and contains .pth files (kernels).
    
    The function iterates through each pair of folders, computes CKA, and saves the results in the output directory.
    If the output directory does not exist, it is created.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    model_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for model_dir1, model_dir2 in itertools.product(model_dirs, repeat=2):
        model1 = os.path.basename(model_dir1)
        model2 = os.path.basename(model_dir2)
        
        
        print(f"Computing CKA between {model1} and {model2}...")
        
        cka_results = compute_cross_model_CKA(model_dir1, model_dir2) 
        
        result_filename = f"{model1}_vs_{model2}.npy"
        result_path = os.path.join(output_dir, result_filename)
        np.save(result_path, cka_results)
        
        logging.info("Saved CKA results to %s", result_path)
        print(f"Saved results to {result_path}")

    
    logging.info("CKA computation completed.")

def compute_all_model_CKA_lowmem(root_dir: str, output_dir: str):
    """
    Computes CKA between models in different folders under the root directory.
    Each folder represents a model architecture and contains .pth files (kernels).
    
    The function iterates through each pair of folders, computes CKA, and saves the results in the output directory.
    If the output directory does not exist, it is created.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    model_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for model_dir1, model_dir2 in itertools.product(model_dirs, repeat=2):
        model1 = os.path.basename(model_dir1)
        model2 = os.path.basename(model_dir2)

        result_filename = f"{model1}_vs_{model2}.npy"
        result_path = os.path.join(output_dir, result_filename)
        
        if os.path.exists(result_path):
            print(f"CKA result between {model1} and {model2} already exists. Skipping computation.")
            continue  # Skip this model pair
        
        print(f"Computing CKA between {model1} and {model2}...")

        cka_results = compute_cross_model_CKA_lowmem(model_dir1, model_dir2) 
        
        np.save(result_path, cka_results)
        logging.info("Saved CKA results to %s", result_path)
        print(f"Saved results to {result_path}", flush=True)

    logging.info("CKA computation completed.")
 


def compute_cross_model_CKA(model_dir1:str,model_dir2:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type1_kernels = []  
    model_type2_kernels = []  
    model_name1 = ""
    model_name2 = ""
    print("model1:",model_dir1)
    print("model2:",model_dir2)

    for filename in os.listdir(model_dir1):
        if not filename.endswith('.pth'):
            continue
        model_path = os.path.join(model_dir1, filename)
        kernel = torch.load(model_path, map_location=device)  
        model_type1_kernels.append(kernel) 
        try:
            _, model_name1 = model_dir1.rsplit('/', 1)
        except Exception as e:
            print(e)
            print("trying different slash")
            _, model_name1 = model_dir1.rsplit('\\',1)
            
    for filename in os.listdir(model_dir2):
        if not filename.endswith('.pth'):
            continue
        model_path = os.path.join(model_dir2, filename)
        kernel = torch.load(model_path, map_location=device)  # Load kernel dictionary onto the device
        try:
            _, model_name2 = model_dir2.rsplit('/', 1)
        except Exception as e:
            print(e)
            print("trying different slash")
            _, model_name2 = model_dir2.rsplit('\\',1)
        model_type2_kernels.append(kernel) 
    cka_results = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))
    layer1_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type1_kernels[0].keys())}
    layer2_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type2_kernels[0].keys())}
    if not os.path.exists("cka_csv"):
        os.makedirs("cka_csv", exist_ok=True)
    panda_data = []
    for i, kernel_A in enumerate(model_type1_kernels):
        cka_inner = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0]))) 
        
        for j, kernel_B in enumerate(model_type2_kernels):
            for layer1, K_x in kernel_A.items():
                for layer2, K_y in kernel_B.items():
                    K_x, K_y = K_x.to(device), K_y.to(device)
                    
                    cka_value = CKA(K_x, K_y)  
                    idx1 = layer1_to_idx[layer1]
                    idx2 = layer2_to_idx[layer2]
                    panda_data.append((layer1,layer2,cka_value))
                    
                    cka_inner[idx1, idx2] += cka_value
                    print(f"CKA({model_name1}.{layer1}, {model_name2}.{layer2}): {cka_value}")
            
        cka_inner /= len(model_type2_kernels)
        cka_results += cka_inner
        
        
        
        print(f"Avg CKA result for kernel {i}: {cka_inner}")

    df = pd.DataFrame([(cka_value[0],cka_value[1],cka_value[2]) for cka_value in panda_data],
                            columns=['Layer1', 'Layer2', 'CKA_Value'])

    df.to_csv(f"cka_csv/CKA_{model_name1}_vs_{model_name2}.csv", sep ="\t", index = False, header = True)

    cka_results /= len(model_type1_kernels)
    return cka_results
    
def compute_cross_model_CKA_lowmem(model_dir1:str,model_dir2:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("found device:",device)

    og_kernels_count = 0
    comp_kernels_count = 0
    found_kernels = 0
    model_name1 = ""
    model_name2 = ""
    print("model1:",model_dir1)
    print("model2:",model_dir2)

    for filename in os.listdir(model_dir1):
        if not filename.endswith('.pth'):
            continue
        model_path = os.path.join(model_dir1, filename)
        og_kernel = torch.load(model_path, map_location=device,weights_only=False)  # Load kernel dictionary onto the device
        try:
            _, model_name1 = model_dir1.rsplit('/', 1)
        except Exception as e:
            print(e)
            print("trying different slash")
            _, model_name1 = model_dir1.rsplit('\\',1)
            
    for filename in os.listdir(model_dir2):
        if not filename.endswith('.pth'):
            continue
        model_path = os.path.join(model_dir2, filename)
        comp_kernel = torch.load(model_path, map_location=device,weights_only=False)  # Load kernel dictionary onto the device
        found_kernels +=1
        try:
            _, model_name2 = model_dir2.rsplit('/', 1)
        except Exception as e:
            print(e)
            print("trying different slash")
            _, model_name2 = model_dir2.rsplit('\\',1)

    cka_results = np.zeros((len(og_kernel), len(comp_kernel)))
    layer1_to_idx = {layer_name: idx for idx, layer_name in enumerate(og_kernel.keys())}
    layer2_to_idx = {layer_name: idx for idx, layer_name in enumerate(comp_kernel.keys())}
    if not os.path.exists("cka_csv"):
        os.makedirs("cka_csv", exist_ok=True)
    panda_data = []

    for filename in os.listdir(model_dir1):
        if not filename.endswith('.pth'):
            continue
        model_path1 = os.path.join(model_dir1, filename)
        og_kernel = torch.load(model_path1, map_location=device,weights_only=False)  
        og_kernels_count +=1
        cka_inner = np.zeros((len(og_kernel), len(comp_kernel))) 
        for filename in os.listdir(model_dir2):
            if not filename.endswith('.pth'):
                continue
            model_path2 = os.path.join(model_dir2, filename)
            comp_kernel = torch.load(model_path2, map_location=device,weights_only=False)  # Load kernel dictionary onto the device
            comp_kernels_count +=1
            for layer1, K_x in og_kernel.items():
                for layer2, K_y in comp_kernel.items():
                    K_x, K_y = K_x.to(device), K_y.to(device)
                    
                    cka_value = CKA(K_x, K_y)  
                    idx1 = layer1_to_idx[layer1]
                    idx2 = layer2_to_idx[layer2]
                    panda_data.append((layer1,layer2,cka_value))
                    
                    cka_inner[idx1, idx2] += cka_value
                    print(f"CKA({model_name1}.{layer1}, {model_name2}.{layer2}): {cka_value}",flush=True)
            
        cka_inner /=found_kernels
        cka_results += cka_inner
        print(f"Avg CKA result for kernel {og_kernels_count}: {cka_inner}",flush=True)

    df = pd.DataFrame([(cka_value[0],cka_value[1],cka_value[2]) for cka_value in panda_data],
                            columns=['Layer1', 'Layer2', 'CKA_Value'])

    df.to_csv(f"cka_csv/CKA_{model_name1}_vs_{model_name2}.csv", sep ="\t", index = False, header = True)

    cka_results /= og_kernels_count
    return cka_results  # Return the CKA similarity matrix



def display_cka_matrix(cka_results, layer_names_model1: list[str], layer_names_model2: list[str],title1:str, title2:str):
    n_layers1 = len(layer_names_model1)
    n_layers2 = len(layer_names_model2)
    matrix = np.zeros((n_layers1, n_layers2))

    for i in range(n_layers1):
        for j in range(n_layers2):
            similarity = cka_results[i, j] 
            matrix[i, j] = np.nan_to_num(similarity)  

    df = pd.DataFrame(matrix, index=layer_names_model1, columns=layer_names_model2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='gist_heat', fmt='.2f', square=True, linewidths=0.5, cbar=True, vmin=0, vmax=1, annot_kws={"size": 20})
    plt.title(f'CKA Similarity Heatmap ({title1} vs {title2} )')
    plt.xlabel(f'{title2}')
    plt.ylabel(f'{title1}')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    
def compute_cka_changes(cka_results):
    diff_matrix = np.zeros_like(cka_results)

    for i in range(1, cka_results.shape[0]):
        diff_matrix[i, i] = abs(cka_results[i, i] - cka_results[i-1, i-1])


    return diff_matrix

def display_differences_matrix_og(cka_results, layer_names_model1: list[str], layer_names_model2: list[str],title1:str, title2:str):
    n_layers1 = len(layer_names_model1)
    n_layers2 = len(layer_names_model2)
    matrix = np.zeros((n_layers1, n_layers2))

    for i in range(n_layers1):
        for j in range(n_layers2):
            similarity = cka_results[i, j]  
            matrix[i, j] = np.nan_to_num(similarity)  

    df = pd.DataFrame(matrix, index=layer_names_model1, columns=layer_names_model2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='gist_heat', fmt='.2f', square=True, linewidths=0.5, cbar=True)
    plt.title(f'CKA Similarity Heatmap ({title1} vs {title2} )')
    plt.xlabel(f'{title2}')
    plt.ylabel(f'{title1}')
    plt.show()
    
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
    
def load_model_metadata(kernel_dir):
    """
    Loads model layer names and formatted model names from JSON files in the kernel directory.
    
    Returns:
        - model_layers: Dict mapping model names to their layer names.
        - model_display_names: Dict mapping model names to formatted display names.
    """
    model_layers = {}
    model_display_names = {}

    for model_name in os.listdir(kernel_dir):
        model_path = os.path.join(kernel_dir, model_name)
        if os.path.isdir(model_path):
            try:
                model_name_without_suffix, _ = model_name.split('_',1)
            except Exception as e:
                print("cannot split. trying without")
                model_name_without_suffix = model_name
            
            json_file_name1 = f'{model_name_without_suffix}_list1.json'
            json_file_name2 = f'{model_name_without_suffix}_list2.json'
            print(f"Looking for file: {json_file_name1}")

            try:
                with open(os.path.join(model_path, json_file_name1), 'r') as f_layer:
                    model_layers[model_name] = json.load(f_layer)

                with open(os.path.join(model_path, json_file_name2), 'r') as f_name:
                    model_display_names[model_name] = json.load(f_name)
            except FileNotFoundError as e:
                print(f"Warning: Missing JSON files for {model_name}. Skipping.")
    return model_layers, model_display_names


def plot_cka_heatmaps(cka_results_dir: str, kernel_dir: str):
    """
    Loads all CKA results from the given directory and visualizes them as heatmaps.
    
    Each row in the final figure represents one model compared to all other models.
    """
    model_layers, model_display_names = load_model_metadata(kernel_dir)
    comparison_files = [f for f in os.listdir(cka_results_dir) if f.endswith('.npy')]

    model_comparisons = {}

    for file in comparison_files:
        parts = file.replace('.npy', '').split('_vs_')
        if len(parts) == 2:
            model1, model2 = parts
            if model1 not in model_comparisons:
                model_comparisons[model1] = []
            if model2 not in model_comparisons:
                model_comparisons[model2] = []
            
            model_comparisons[model1].append((model2, file))
            model_comparisons[model2].append((model1, file))
    print("model_comp:",model_comparisons)
    model_names = sorted(model_comparisons.keys())
    print("model_names:",model_names)
    for i, model1 in enumerate(model_names):
        comparisons = model_comparisons[model1]
        prime_model_len = len(model_layers[model1][0])
        for j, (model2, file) in enumerate(comparisons):
            file_path = os.path.join(cka_results_dir, file)
            cka_matrix = np.load(file_path) 
            print("filepath:",file_path)
            print(cka_matrix)
            title1 = model_display_names.get(model1, model1)
            title2 = model_display_names.get(model2, model2)
            
            if (prime_model_len !=cka_matrix.shape[0]):
                cka_matrix = cka_matrix.transpose()
                
            layers1 = model_layers.get(model1, [f"Layer {i}" for i in range(cka_matrix.shape[0])])[0]
            layers2 = model_layers.get(model2, [f"Layer {i}" for i in range(cka_matrix.shape[1])])[0]
            if isinstance(title1, list):
                title1 = title1[0]
            if isinstance(title2, list):
                title2 = title2[0]
            print(prime_model_len ," vs ", len(layers1), " vs ", len(layers2))
            print(prime_model_len, " vs ", cka_matrix.shape[0])
            
            print(model1)
            print(model2)
            print("display")
            display_cka_matrix(cka_matrix, layers1, layers2, title1, title2,"cka_heatmaps")
            
            
def display_cka_matrix(cka_results, layer_names_model1: list[str], layer_names_model2: list[str], Overall_title:str,title1: str, title2: str, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    n_layers1 = len(layer_names_model1)
    n_layers2 = len(layer_names_model2)
    matrix = np.zeros((n_layers1, n_layers2))

    for i in range(n_layers1):
        for j in range(n_layers2):
            similarity = cka_results[i, j] 
            matrix[i, j] = np.nan_to_num(similarity)  

    df = pd.DataFrame(matrix, index=layer_names_model1, columns=layer_names_model2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='gist_heat', fmt='.2f', square=True, 
                linewidths=0.5, cbar=True, vmin=0, vmax=1,annot_kws={"size": 18,"weight":"bold"})
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.title(f'{Overall_title} ({title1} vs {title2})',fontsize=18)
    plt.xlabel(f'{title2}', fontsize=18)
    plt.ylabel(f'{title1}', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    filename = f"{title1}_vs_{title2}.png".replace(" ", "_") 
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close() 


def compose_heat_matrix(result_folder: str, output_folder: str, title: str = "cka heatmap"):
    """
    Reads CKA results from .npy files in the result_folder, constructs an NxN matrix,
    and generates a heatmap of CKA values, ensuring 'Shallow' appears in the top-right corner.
    
    Args:
        result_folder (str): Path to the folder containing .npy CKA result files.
        output_folder (str): Path to save the generated heatmap image.
        title (str): Title of the heatmap.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cka_files = [f for f in os.listdir(result_folder) if f.endswith(".npy")]
    
    model_names = sorted(set(
        name.split("_vs_")[0] for name in cka_files
    ).union(
        name.split("_vs_")[1].replace(".npy", "") for name in cka_files
    ))  

    num_models = len(model_names)
    cka_matrix = np.zeros((num_models, num_models))
    
    for file in cka_files:
        model1, model2 = file.replace(".npy", "").split("_vs_")
        cka_value = np.load(os.path.join(result_folder, file))[0, 0]  # Extract scalar value
        print("file:",file)
        print("model1:",model1, "model2:",model2, "cka value:",cka_value)
        i, j = model_names.index(model1), model_names.index(model2)
        cka_matrix[i, j] = cka_value
        cka_matrix[j, i] = cka_value 

    cka_matrix = np.flipud(cka_matrix)
    model_names_reversed = list(reversed(model_names))

    df = pd.DataFrame(cka_matrix, index=model_names_reversed, columns=model_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='gist_heat', fmt='.2f', square=True, linewidths=0.5, cbar=True, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Model')
    
    filepath = os.path.join(output_folder, f"{title}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Heatmap saved to {filepath}")


def compose_heat_matrix_shared(result_folder: str, output_folder: str, csv_folder: str, title: str = "cka heatmap"):
    os.makedirs(output_folder, exist_ok=True)

    model_name_map = {
        "ShallowFBCSPNet": "Shallow",
        "ShallowRNNNet": "RNN",
        "ShallowLSTM": "LSTM",
        "ShallowAttentionNet": "Attention",
        "ShallowSGCNNet": "SGCN"
    }

    model_unanimous_map = {
        "ShallowFBCSPNet": 10726,
        "ShallowRNNNet": 4832,
        "ShallowLSTM": 2238,
        "ShallowAttentionNet": 9387,
        "ShallowSGCNNet": 11027
    }

    cka_files = [f for f in os.listdir(result_folder) if f.endswith(".npy")]

    model_names = sorted(set(
        name.split("_vs_")[0] for name in cka_files
    ).union(
        name.split("_vs_")[1].replace(".npy", "") for name in cka_files
    ))
    print(model_names)
    def reorder(models):
        return ['ShallowFBCSP'] + [m for m in models if m not in {'ShallowFBCSPNet', 'ShallowFBCSP'}]

    model_names= reorder(model_names)
    print(model_names)
    num_models = len(model_names)
    print(num_models)
    cka_matrix = np.zeros((num_models, num_models))
    annotation_matrix = [["" for _ in range(num_models)] for _ in range(num_models)]

    for file in cka_files:
        model1, model2 = file.replace(".npy", "").split("_vs_")
        cka_value = np.load(os.path.join(result_folder, file))[0, 0]
        print("file:",file)
        print("model1:",model1, "model2:",model2, "cka value:",cka_value)
        if model1 not in model_names or model2 not in model_names:
            continue
        i, j = model_names.index(model1), model_names.index(model2)
        cka_matrix[i, j] = cka_value
        cka_matrix[j, i] = cka_value  # symmetry

        model1_csv = model_name_map.get(model1, model1)
        model2_csv = model_name_map.get(model2, model2)

        keyfile1 = os.path.join(csv_folder, f"Shared_Keys_{model1_csv}_and_{model2_csv}.csv")
        keyfile2 = os.path.join(csv_folder, f"Shared_Keys_{model2_csv}_and_{model1_csv}.csv")
        keyfile = keyfile1 if os.path.exists(keyfile1) else keyfile2 if os.path.exists(keyfile2) else None
        print("keyfile:",keyfile)
        if keyfile and os.path.isfile(keyfile):
            with open(keyfile, "r") as f:
                shared_lines = f.readlines()
            num_keys = len(shared_lines) - 1 if ("idx" in shared_lines[0].lower() or "index" in shared_lines[0].lower()) else len(shared_lines)
        else:
            num_keys = 0
        if model1 == model2:
            num_keys = model_unanimous_map.get(model1, 0)

        print(f"num keys: {num_keys}, for models: {model1} and {model2}")

        annotation_text = f"{cka_value:.2f}"
        annotation_matrix[i][j] = annotation_text
        annotation_matrix[j][i] = annotation_text

    cka_matrix = np.flipud(cka_matrix)
    print(cka_matrix)
    annotation_matrix = list(reversed(annotation_matrix))
    model_names = sorted(name[:-3] if name.endswith("Net") else name for name in model_names)

    model_names = reorder(model_names)
    model_names = [name.removeprefix("Shallow").removesuffix("Net") for name in model_names]

    model_names_reversed = list(reversed(model_names))
    print(model_names_reversed)
    df = pd.DataFrame(cka_matrix, index=model_names_reversed, columns=model_names)
    annot_df = pd.DataFrame(annotation_matrix, index=model_names_reversed, columns=model_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df,
        annot=annot_df,
        fmt='',
        cmap='gist_heat',
        square=True,
        linewidths=0.5,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={"size": 24, "weight": "bold"}
    )
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    plt.title(title, fontsize=22)
    # plt.xlabel('Model', fontsize=18)
    # plt.ylabel('Model', fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    filepath = os.path.join(output_folder, f"{title}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Heatmap saved to {filepath}")



def compose_heat_matrix_acc(result_folder: str, output_folder: str, model_path: str, title: str = "cka heatmap"):

    os.makedirs(output_folder, exist_ok=True)

    model_name_map = {
        "ShallowFBCSP": "ShallowFBCSP",
        "ShallowRNN": "ShallowRNN",
        "ShallowLSTM": "ShallowLSTM",
        "ShallowAttention": "ShallowAtt",
        "ShallowSGCN": "ShallowSGCN"
    }

    cka_files = [f for f in os.listdir(result_folder) if f.endswith(".npy")]

    model_names_raw = set(
        name.split("_vs_")[0] for name in cka_files
    ).union(
        name.split("_vs_")[1].replace(".npy", "") for name in cka_files
    )
    model_names = sorted(name[:-3] if name.endswith("Net") else name for name in model_names_raw)

    model_accuracies = {}
    for model_name in model_names:
        model_name_folder = model_name_map.get(model_name, model_name)
        model_folder = os.path.join(model_path, model_name_folder)
        model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth") and 'state' not in f.lower()]
        accuracies = []

        for model_file in model_files:
            parts = model_file.replace(".pth", "").split("_")
            try:
                acc = float(parts[1])
                accuracies.append(acc)
            except:
                print(f"Warning: could not parse accuracy from {model_file}")

        avg_acc = np.mean(accuracies) if accuracies else 0.0
        model_accuracies[model_name] = avg_acc
        print(f"Model {model_name}: Average Accuracy = {avg_acc:.2f}%")

    def reorder(models):
        return ['ShallowFBCSP'] + [m for m in models if m != 'ShallowFBCSP']
    model_names = reorder(model_names)
    model_accuracies = {k: model_accuracies[k] for k in model_names}


    num_models = len(model_names)
    matrix_vals = np.zeros((num_models, num_models))
    annotations = [["" for _ in range(num_models)] for _ in range(num_models)]

    for file in cka_files:
        raw_model1, raw_model2 = file.replace(".npy", "").split("_vs_")
        model1 = raw_model1[:-3] if raw_model1.endswith("Net") else raw_model1
        model2 = raw_model2[:-3] if raw_model2.endswith("Net") else raw_model2

        if model1 not in model_names or model2 not in model_names:
            continue

        i, j = model_names.index(model1), model_names.index(model2)
        cka_value = np.load(os.path.join(result_folder, file))[0, 0]

        matrix_vals[i][j] = cka_value
        matrix_vals[j][i] = cka_value

        acc1 = model_accuracies[model1]
        acc2 = model_accuracies[model2]

        annotation = f"{cka_value:.2f}"
        annotations[i][j] = annotation
        annotations[j][i] = f"{cka_value:.2f}"

    matrix_vals = np.flipud(matrix_vals)
    annotations = annotations[::-1]

    y_labels = list(reversed([f"{name.removeprefix("Shallow").removesuffix("Net")}\n({model_accuracies[name]:.2f}%)" for name in model_names]))
    x_labels = [f"{name.removeprefix("Shallow").removesuffix("Net")}\n({model_accuracies[name]:.2f}%)" for name in model_names]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix_vals,
        annot=annotations,
        fmt='',
        cmap='gist_heat',
        square=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.5,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={"size": 24, "weight": "bold"} 
    )
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    plt.title(title, fontsize=22)
    # plt.xlabel('Model (Avg Accuracy)', fontsize=18)
    # plt.ylabel('Model (Avg Accuracy)', fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    filepath = os.path.join(output_folder, f"{title}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Heatmap saved to {filepath}")
    
    
def adjacency_matrix_motion():
    grid = np.array([
        [ 0,  0,  0,  1,  0,  0,  0],
        [ 0,  2,  3,  4,  5,  6,  0],
        [ 7,  8,  9, 10, 11, 12, 13],
        [ 0, 14, 15, 16, 17, 18,  0],
        [ 0,  0, 19, 20, 21,  0,  0],
        [ 0,  0,  0, 22,  0,  0,  0]
    ])

    positions = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0:
                positions[grid[i, j]] = (i, j)

    n = len(positions)
    adj_matrix = np.zeros((n, n), dtype=int)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # up, down, left, right
                (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonals

    for elec, (x, y) in positions.items():
        for dx, dy in directions:
            neighbor_x, neighbor_y = x + dx, y + dy
            if (neighbor_x, neighbor_y) in positions.values():
                neighbor_elec = [k for k, v in positions.items() if v == (neighbor_x, neighbor_y)][0]
                adj_matrix[elec - 1, neighbor_elec - 1] = 1  
                adj_matrix[neighbor_elec - 1, elec - 1] = 1 

    return adj_matrix, positions  


def adjacency_matrix_FACED():
    grid = np.array([
        [0,0,0,0,1,0,2,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,6,0,4,0,3,0,5,0,7,0],
        [0,0,10,0,8,0,9,0,11,0,0],
        [17,15,0,13,0,12,0,14,0,16,18],
        [0,0,21,0,19,0,20,0,22,0,0],
        [0,26,0,24,0,23,0,25,0,27,0],
        [0,0,0,0,28,0,29,0,0,0,0],
        [0,0,0,0,31,30,32,0,0,0,0]
    ])

    positions = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0:
                positions[grid[i, j]] = (i, j)

    n = len(positions)
    adj_matrix = np.zeros((n, n), dtype=int)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # up, down, left, right
                (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonals

    for elec, (x, y) in positions.items():
        for dx, dy in directions:
            neighbor_x, neighbor_y = x + dx, y + dy
            if (neighbor_x, neighbor_y) in positions.values():
                neighbor_elec = [k for k, v in positions.items() if v == (neighbor_x, neighbor_y)][0]
                adj_matrix[elec - 1, neighbor_elec - 1] = 1  
                adj_matrix[neighbor_elec - 1, elec - 1] = 1 

    return adj_matrix, positions 

def adjacency_matrix_distance_motion(positions, delta=1.0):
    """Computes the full distance matrix and symmetric adjacency weights."""
    n = len(positions)
    distance_matrix = np.zeros((n, n))
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x1, y1 = positions[i + 1]
            x2, y2 = positions[j + 1]
            distance_matrix[i, j] = max(abs(x1 - x2), abs(y1 - y2))

    for i in range(n):
        for j in range(i, n):  
            if i != j:  
                d_ij = distance_matrix[i, j] * 3.5 
                adj_matrix[i, j] = min(1, delta / (d_ij ** 2)) if d_ij > 0 else 0
                adj_matrix[j, i] = adj_matrix[i, j]  
                
                
    
    return distance_matrix, adj_matrix
            
def adjacency_matrix_distance_FACED(positions, delta=1.0):
    """Computes the full distance matrix and symmetric adjacency weights."""
    n = len(positions)
    distance_matrix = np.zeros((n, n))
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x1, y1 = positions[i + 1]
            x2, y2 = positions[j + 1]
            distance_matrix[i, j] = max(abs(x1 - x2), abs(y1 - y2))

    for i in range(n):
        for j in range(i, n):  
            if i != j:  
                d_ij = distance_matrix[i, j] * 3.5  # Convert to physical distance
                adj_matrix[i, j] = min(1, delta / (d_ij ** 2)) if d_ij > 0 else 0
                adj_matrix[j, i] = adj_matrix[i, j]  # Enforce symmetry
    
    return distance_matrix, adj_matrix
            