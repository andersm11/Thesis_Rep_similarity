import torch
import pandas as pd
from torch.utils.data import Subset
# Load your full dataset
X = torch.load('../Datasets/FACED_dataset/emotion_test_set.pt')

# Load indices from CSV and convert to int
spatial_indices = pd.read_csv("../Shared_Keys/Shared_Keys_Spatial.csv",skiprows=1, header=None)[0].astype(int).tolist()

print("Spatial Indices:", spatial_indices)
temporal_indices = pd.read_csv("../Shared_Keys/Shared_Keys_Temporal.csv",skiprows=1, header=None)[0].astype(int).tolist()

# Create subsets
spatial_subset = Subset(X, spatial_indices)
temporal_subset = Subset(X, temporal_indices)

# Print the first 20 samples from spatial subset
print("First 20 Samples from Spatial Subset:")
for i in range(min(20, len(spatial_subset))):
    sample = spatial_subset[i]
    print(f"[{i}] Original Index {spatial_indices[i]}: {sample}")

print("\n" + "=" * 80 + "\n")

# Print the first 20 samples from temporal subset
print("First 20 Samples from Temporal Subset:")
for i in range(min(20, len(temporal_subset))):
    sample = temporal_subset[i]
    #print(f"[{i}] Original Index {temporal_indices[i]}: {sample}")
