from imports import *
import json
import os
import itertools
import numpy as np
import logging
import math

# CKA math

def linear_kernel(X):
    """Computes the linear kernel matrix for X."""
    return torch.matmul(X,X.T)  # Dot product

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


def extract_model_activations(model: torch.nn.Module, input_tensor: torch.Tensor, output_dir: str, layer_names: list[str], batch_size: int = 128):
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

    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[i:i + batch_size]  # Select current batch
            _ = model(batch)  # Forward pass through the model

            # Save activations after each batch
            for name, activation in activations.items():
                batch_idx = i // batch_size + 1  
                print(f"saving: {name}_batch_{batch_idx}.pt")
                torch.save(activation, os.path.join(output_dir, f"{name}_batch_{batch_idx}.pt"))
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


import os
import torch

def compute_full_kernels(layer_names: list[str], total_nr_batches: int, batch_size: int, total_samples: int, load_dir: str, id:str,save_dir: str, n_batches: int = 1, use_cuda: bool = False):
    """Computes the kernels for model and saves them to the given directory if not already saved.
    Deletes all elements in load_dir after computation and adds an empty 'done' file."""
    
    save = save_dir+f"/{id}"
    if os.path.exists(save):
        print(f"Kernel file already exists at {save}. Skipping computation.")
        return
    
    model_kernels = {}
    
    for layer in layer_names:
        print("Got layer:", layer)
        kernel = compute_kernel_full_lowmem(layer, total_nr_batches, batch_size, total_samples, load_dir, n_batches, use_cuda)
        
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
    use_cuda: bool = False
):
    """Computes kernels for multiple models and saves them in the given directory."""
    model_files = os.listdir(models_directory)
    total_samples = input_data.shape[0]
    total_nr_batches = math.ceil(total_samples / batch_size)
    final_layer_names=[]
    final_model_names=[]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    for model_file in model_files:
        if not model_file.endswith('state.pth'):
            model_name, loss, seed = model_file.rsplit('_', 2)
            model = load_model(model_file, models_directory)
            print(model)
            model.to(device)
            model.eval()
            
            activation_dir = os.path.join(activations_root_directory, model_name, f"{loss}_{seed}")
            try:
                found_names = extract_model_activations(model, input_data, activation_dir, layer_names, batch_size=batch_size)
            except Exception as e:
                print(f"Error extracting activations: {e}. Resampling input data to 1000 samples and retrying.")
                downsampled_data = F.interpolate(input_data, size=(1000,), mode='linear', align_corners=False)
                found_names = extract_model_activations(model, downsampled_data, activation_dir, layer_names, batch_size=batch_size)
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
                use_cuda
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
    use_cuda: bool = False
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
        "use_cuda": use_cuda
    }
    for direc in model_directories:
        model_path = os.path.join(target_directory,direc)
        layer_list,model_list = compute_multi_model_kernels(model_path, **kwargs)
        print(layer_list)
        print("model list: ",model_list)
        model_name = model_list[0]
        kernel_specific_path = os.path.join(kernels_directory,model_name)
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
    # Set device to GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if len(model_dirs) != 2:
        raise ValueError("Expected exactly two model type directories in the root directory.")

    model_type1_kernels = []  # For the first model type
    model_type2_kernels = []  # For the second model type
    model_name1 = ""
    model_name2 = ""

    for i, model_dir in enumerate(model_dirs):
        for filename in os.listdir(model_dir):
            model_path = os.path.join(model_dir, filename)
            kernel = torch.load(model_path, map_location=device)  # Load kernel dictionary onto the device

            if i == 0:
                model_type1_kernels.append(kernel)  # For the first model type
                _, model_name1 = model_dir.rsplit('/', 1)
            else:
                _, model_name2 = model_dir.rsplit('/', 1)
                model_type2_kernels.append(kernel)  # For the second model type

    cka_results = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))

    layer1_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type1_kernels[0].keys())}
    layer2_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type2_kernels[0].keys())}

    for i, kernel_A in enumerate(model_type1_kernels):
        cka_inner = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))  # Initialize inner CKA matrix for each kernel_A
        
        for j, kernel_B in enumerate(model_type2_kernels):
            for layer1, K_x in kernel_A.items():
                for layer2, K_y in kernel_B.items():
                    # Move kernels to GPU if available
                    K_x, K_y = K_x.to(device), K_y.to(device)
                    
                    cka_value = CKA(K_x, K_y)  # Compute CKA between this pair of kernels
                    idx1 = layer1_to_idx[layer1]
                    idx2 = layer2_to_idx[layer2]
                    
                    cka_inner[idx1, idx2] += cka_value
                    print(f"CKA({model_name1}.{layer1}, {model_name2}.{layer2}): {cka_value}")
            
        cka_inner /= len(model_type2_kernels)
        print(len(model_type2_kernels))
        cka_results += cka_inner
        
        print(f"Avg CKA result for kernel {i}: {cka_inner}")

    cka_results /= len(model_type1_kernels)
    print(len(model_type1_kernels))
    return cka_results  # Return the CKA similarity matrix


def compute_all_model_CKA(root_dir: str, output_dir: str):
    """
    Computes CKA between models in different folders under the root directory.
    Each folder represents a model architecture and contains .pth files (kernels).
    
    The function iterates through each pair of folders, computes CKA, and saves the results in the output directory.
    If the output directory does not exist, it is created.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(output_dir, 'cka_computation.log')
    logging.basicConfig(filename=log_file, 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Starting CKA computation for models in %s", root_dir)
    
    # Get all folders in the root directory
    model_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Iterate over all unique pairs of model directories
    for model_dir1, model_dir2 in itertools.product(model_dirs, repeat=2):
        model1 = os.path.basename(model_dir1)
        model2 = os.path.basename(model_dir2)
        
        logging.info("Computing CKA between %s and %s", model1, model2)
        
        print(f"Computing CKA between {model1} and {model2}...")
        
        # Compute cross-model CKA
        cka_results = compute_cross_model_CKA(model_dir1, model_dir2)  # Assumed function
        
        # Save results to a file in the output directory
        result_filename = f"{model1}_vs_{model2}.npy"
        result_path = os.path.join(output_dir, result_filename)
        np.save(result_path, cka_results)
        
        logging.info("Saved CKA results to %s", result_path)
        print(f"Saved results to {result_path}")
    
    logging.info("CKA computation completed.")
 


def compute_cross_model_CKA(model_dir1:str,model_dir2:str):
    # Set device to GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type1_kernels = []  # For the first model type
    model_type2_kernels = []  # For the second model type
    model_name1 = ""
    model_name2 = ""
    print(model_dir1)
    print(model_dir2)

    for filename in os.listdir(model_dir1):
        if not filename.endswith('.pth'):
            continue
        model_path = os.path.join(model_dir1, filename)
        print(model_path)
        kernel = torch.load(model_path, map_location=device)  # Load kernel dictionary onto the device
        model_type1_kernels.append(kernel)  # For the first model type
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
        print(model_path)
        kernel = torch.load(model_path, map_location=device)  # Load kernel dictionary onto the device
        try:
            _, model_name2 = model_dir2.rsplit('/', 1)
        except Exception as e:
            print(e)
            print("trying different slash")
            _, model_name2 = model_dir2.rsplit('\\',1)
        model_type2_kernels.append(kernel)  # For the second model type

    cka_results = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))

    layer1_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type1_kernels[0].keys())}
    layer2_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type2_kernels[0].keys())}

    for i, kernel_A in enumerate(model_type1_kernels):
        cka_inner = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))  # Initialize inner CKA matrix for each kernel_A
        
        for j, kernel_B in enumerate(model_type2_kernels):
            for layer1, K_x in kernel_A.items():
                for layer2, K_y in kernel_B.items():
                    # Move kernels to GPU if available
                    K_x, K_y = K_x.to(device), K_y.to(device)
                    
                    cka_value = CKA(K_x, K_y)  # Compute CKA between this pair of kernels
                    idx1 = layer1_to_idx[layer1]
                    idx2 = layer2_to_idx[layer2]
                    
                    cka_inner[idx1, idx2] += cka_value
                    print(f"CKA({model_name1}.{layer1}, {model_name2}.{layer2}): {cka_value}")
            
        cka_inner /= len(model_type2_kernels)
        print(len(model_type2_kernels))
        cka_results += cka_inner
        
        print(f"Avg CKA result for kernel {i}: {cka_inner}")

    cka_results /= len(model_type1_kernels)
    print(len(model_type1_kernels))
    return cka_results  # Return the CKA similarity matrix
    


def display_cka_matrix(cka_results, layer_names_model1: list[str], layer_names_model2: list[str],title1:str, title2:str):
    n_layers1 = len(layer_names_model1)
    n_layers2 = len(layer_names_model2)
    matrix = np.zeros((n_layers1, n_layers2))

    for i in range(n_layers1):
        for j in range(n_layers2):
            similarity = cka_results[i, j]  # Access similarity directly from the ndarray
            matrix[i, j] = np.nan_to_num(similarity)  # Handle NaN or Inf values

    df = pd.DataFrame(matrix, index=layer_names_model1, columns=layer_names_model2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='gist_heat', fmt='.2f', square=True, linewidths=0.5, cbar=True, vmin=0, vmax=1)
    plt.title(f'CKA Similarity Heatmap ({title1} vs {title2} )')
    plt.xlabel(f'{title2}')
    plt.ylabel(f'{title1}')
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

    # Plot the heatmap
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
            # Remove '_model' from model_name if it's in the name
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
    # Load metadata (layer names & model display names)
    model_layers, model_display_names = load_model_metadata(kernel_dir)
    # Collect available comparisons
    comparison_files = [f for f in os.listdir(cka_results_dir) if f.endswith('.npy')]

    # Extract unique model names and group comparisons by model
    model_comparisons = {}

    for file in comparison_files:
        parts = file.replace('.npy', '').split('_vs_')
        if len(parts) == 2:
            model1, model2 = parts
            # Group comparisons for each model
            if model1 not in model_comparisons:
                model_comparisons[model1] = []
            if model2 not in model_comparisons:
                model_comparisons[model2] = []
            
            model_comparisons[model1].append((model2, file))
            model_comparisons[model2].append((model1, file))
    print("model_comp:",model_comparisons)
    # Sort the model names for consistent ordering
    model_names = sorted(model_comparisons.keys())
    print("model_names:",model_names)
    # Plot heatmaps for each model comparison, one row at a time
    for i, model1 in enumerate(model_names):
        comparisons = model_comparisons[model1]
        prime_model_len = len(model_layers[model1][0])
        # Plot heatmaps for each comparison
        for j, (model2, file) in enumerate(comparisons):
            file_path = os.path.join(cka_results_dir, file)
            cka_matrix = np.load(file_path)  # Load CKA result
            print("filepath:",file_path)
            print(cka_matrix)
            # Get model display names and layers
            title1 = model_display_names.get(model1, model1)
            title2 = model_display_names.get(model2, model2)
            
            if (prime_model_len !=cka_matrix.shape[0]):
                cka_matrix = cka_matrix.transpose()
                
            layers1 = model_layers.get(model1, [f"Layer {i}" for i in range(cka_matrix.shape[0])])[0]
            layers2 = model_layers.get(model2, [f"Layer {i}" for i in range(cka_matrix.shape[1])])[0]
            # Ensure they are strings, not lists
            if isinstance(title1, list):
                title1 = title1[0]
            if isinstance(title2, list):
                title2 = title2[0]
            # Display CKA matrix using the helper function
            #print(title1)
            print(prime_model_len ," vs ", len(layers1), " vs ", len(layers2))
            print(prime_model_len, " vs ", cka_matrix.shape[0])
            
            print(model1)
            print(model2)
            print("display")
            display_cka_matrix(cka_matrix, layers1, layers2, title1, title2,"cka_heatmaps")
            
            
def display_cka_matrix(cka_results, layer_names_model1: list[str], layer_names_model2: list[str], title1: str, title2: str, output_folder):
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
    sns.heatmap(df, annot=True, cmap='gist_heat', fmt='.2f', square=True, linewidths=0.5, cbar=True, vmin=0, vmax=1)
    plt.title(f'CKA Similarity Heatmap ({title1} vs {title2})')
    plt.xlabel(f'{title2}')
    plt.ylabel(f'{title1}')

    filename = f"{title1}_vs_{title2}.png".replace(" ", "_") 
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close() 


def compose_heat_matrix(result_folder: str, output_folder: str,title:str="cka heatmap"):
    """
    Reads CKA results from .npy files in the result_folder, constructs an NxN matrix,
    and generates a heatmap of CKA values.
    
    Args:
        result_folder (str): Path to the folder containing .npy CKA result files.
        output_folder (str): Path to save the generated heatmap image.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Read all .npy files in the folder
    cka_files = [f for f in os.listdir(result_folder) if f.endswith(".npy")]
    
    # Extract unique model names from filenames
    model_names = sorted(set(
        name.split("_vs_")[0] for name in cka_files
    ).union(
        name.split("_vs_")[1].replace(".npy", "") for name in cka_files
    ))  # Maintain normal order for bottom-left to top-right diagonal alignment
    
    # Initialize an NxN matrix
    num_models = len(model_names)
    cka_matrix = np.zeros((num_models, num_models))
    
    # Fill the matrix with CKA values
    for file in cka_files:
        model1, model2 = file.replace(".npy", "").split("_vs_")
        cka_value = np.load(os.path.join(result_folder, file))[0, 0]  # Extract scalar value
        i, j = model_names.index(model1), model_names.index(model2)
        cka_matrix[i, j] = cka_value
        cka_matrix[j, i] = cka_value  # Ensure symmetry
    
    # Flip matrix to align diagonal from bottom-left to top-right
    cka_matrix = np.flipud(cka_matrix)
    model_names_reversed = list(reversed(model_names))
    
    # Create heatmap
    df = pd.DataFrame(cka_matrix, index=model_names_reversed, columns=model_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='gist_heat', fmt='.2f', square=True, linewidths=0.5, cbar=True, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Model')
    
    # Save heatmap
    filepath = os.path.join(output_folder, f"{title}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Heatmap saved to {filepath}")
    
    
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


def adjacency_matrix_distance_motion(positions, delta=1.0):
    """Computes the full distance matrix and symmetric adjacency weights."""
    n = len(positions)
    distance_matrix = np.zeros((n, n))
    adj_matrix = np.zeros((n, n))

    # Compute Euclidean distances between ALL pairs of electrodes
    for i in range(n):
        for j in range(n):
            x1, y1 = positions[i + 1]
            x2, y2 = positions[j + 1]
            distance_matrix[i, j] = max(abs(x1 - x2), abs(y1 - y2))

    # Compute physical distance and adjacency weights, enforce symmetry
    for i in range(n):
        for j in range(i, n):  # Only compute upper triangular matrix (i <= j)
            if i != j:  # Ignore self-connections
                d_ij = distance_matrix[i, j] * 3.5  # Convert to physical distance
                adj_matrix[i, j] = min(1, delta / (d_ij ** 2)) if d_ij > 0 else 0
                adj_matrix[j, i] = adj_matrix[i, j]  # Enforce symmetry
    
    return distance_matrix, adj_matrix
            
