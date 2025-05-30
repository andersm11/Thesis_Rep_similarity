import performance_functions
from CKA_functions import fix_dataset_shape,load_dataset
from torch.utils.data import DataLoader
from torch import argmax
import torch
from performance_functions import *

X = torch.load('FACED_dataset/emotion_test_set.pt')
test_loader = DataLoader(X, batch_size=16)
compute_all_model_confusion(test_loader,"models","all_model_conf") 
