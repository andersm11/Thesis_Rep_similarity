import torch
import pandas as pd
from torch.utils.data import Subset
# Load your full dataset
X = torch.load('../Datasets/FACED_dataset/emotion_test_set.pt')


print("Spatial Indices:", spatial_indices)
all_indices = pd.read_csv("../Shared_Keys/Shared_Keys_all.csv",skiprows=1, header=None)[0].astype(int).tolist()


all_subset = Subset(X, all_indices)

