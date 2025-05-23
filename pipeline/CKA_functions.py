from imports import *
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
def extract_model_activations(model: torch.nn.Module, input_tensor: torch.Tensor, output_dir: str, layer_names: list[str], batch_size: int = 128):
    found_layer_names = []
    # Check if the output directory exists and is not empty
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



#compute kernels for all models in directory
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
    for model_file in model_files:
        if not model_file.endswith('state.pth'):
            model_name, loss, seed = model_file.rsplit('_', 2)
            model = load_model(model_file, models_directory)
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

    model_type1_kernels = []  # For the first model type
    model_type2_kernels = []  # For the second model type
    model_name1 =""
    model_name2 = ""
    for i, model_dir in enumerate(model_dirs):
        for filename in os.listdir(model_dir):
            model_path = os.path.join(model_dir, filename)
            kernel = torch.load(model_path)  # Load kernel dictionary

            if i == 0:
                model_type1_kernels.append(kernel)  # For the first model type
                _ , model_name1 = model_dir.rsplit('/',1)
            else:
                _ , model_name2 = model_dir.rsplit('/',1)
                model_type2_kernels.append(kernel)  # For the second model type

    cka_results = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))

    layer1_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type1_kernels[0].keys())}
    layer2_to_idx = {layer_name: idx for idx, layer_name in enumerate(model_type2_kernels[0].keys())}

    for i, kernel_A in enumerate(model_type1_kernels):
        cka_inner = np.zeros((len(model_type1_kernels[0]), len(model_type2_kernels[0])))  # Initialize inner CKA matrix for each kernel_A
        
        for j, kernel_B in enumerate(model_type2_kernels):
            for layer1, K_x in kernel_A.items():
                for layer2, K_y in kernel_B.items():
                    cka_value = CKA(K_x, K_y)  # Compute CKA between this pair of kernels
                    idx1 = layer1_to_idx[layer1]
                    idx2 = layer2_to_idx[layer2]
                    
                    cka_inner[idx1, idx2] += cka_value
                    print(f"CKA({model_name1}.{layer1}, {model_name2}.{layer2}): {cka_value}")
            
        cka_inner /= len(model_type2_kernels)
        
        cka_results += cka_inner
        
        print(f"Avg CKA result for kernel {i}: {cka_inner}")

    cka_results /= len(model_type1_kernels)

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

def display_differences_matrix(cka_results, layer_names_model1: list[str], layer_names_model2: list[str],title1:str, title2:str):
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
            

            
