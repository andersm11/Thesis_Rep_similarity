from CKA_functions import compose_heat_matrix_acc,compose_heat_matrix_shared_full,compose_heat_matrix_shared,display_cka_matrix
import os
import numpy as np
compose_heat_matrix_shared("../cka_results","motion_cka","../Shared_Keys_motion","CKA Shared Classification Temporal (Motion)")
#compose_heat_matrix_acc("../cka_results","motion_cka","../motion_models","CKA Classification Temporal (Motion)")

full = 0


# Path to the directory containing the .npy files
npy_dir = "../cka_results"

# Lookup table for model-specific layer names
layer_name_lookup = {
    "ShallowFBCSPNet": ["temporal","spatial", "pool", "fc"],
    "ShallowAttentionNet": ["temporal", "spatial_att", "pool", "fc"],
    "ShallowSGCNNet": ["temporal", "sgconv", "pool", "fc"],
    "ShallowLSTM": ["lstm", "spatial", "pool", "fc"],
    "ShallowRNNNet": ["RNN", "spatial", "pool", "fc"],
}
# layer_name_lookup = {
#     "ShallowFBCSPNet": ["temporal", "fc"],
#     "ShallowAttentionNet": ["temporal", "spatial_att", "pool", "fc"],
#     "ShallowSGCNNet": ["temporal", "sgconv", "pool", "fc"],
#     "ShallowLSTM": ["lstm" ,"fc"],
#     "ShallowRNNNet": ["RNN", "fc"],
# }

# List all .npy files in the directory
if full != 0:
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} .npy files in {npy_dir}.")
    # Load and display the contents of each .npy file
    for npy_file in npy_files:
        # Parse model names from filename
        base_name = os.path.splitext(npy_file)[0]
        try:
            model1, model2 = base_name.split("_vs_")
        except ValueError:
            print(f"Skipping invalid filename: {npy_file}")
            continue

        # Look up layer names
        layer_names1 = layer_name_lookup.get(model1)
        layer_names2 = layer_name_lookup.get(model2)

        if layer_names1 is None or layer_names2 is None:
            print(f"Missing layer name lookup for: {model1} or {model2}")
            continue

        # Load data
        file_path = os.path.join(npy_dir, npy_file)
        data = np.load(file_path, allow_pickle=True)

        # Display matrix
        display_cka_matrix(data, layer_names1, layer_names2,"CKA Temporal Consensus", model1, model2, "cka_heatmaps/Temporal_consensus")

        # Print the raw data
        print(f"Contents of {npy_file}:")
        print(data)
        print("-" * 60)
