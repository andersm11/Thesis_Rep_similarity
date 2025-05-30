import numpy as np
import os
import torch
import json
import pandas as pd
from Spatial_attention_FACED import ShallowAttentionNet
from SGCN_FACED_norm import ShallowSGCNNet
from shallow_laurits_faced import ShallowFBCSPNet
from RNN_model import ShallowRNNNet
from LSTM_model import ShallowLSTMNet
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from CKA_functions import adjacency_matrix_FACED, adjacency_matrix_distance_FACED


def get_e_index(dm):
    threshold = 0
    source_nodes = []
    target_nodes = []
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            if dm[i, j] >= threshold:
                source_nodes.append(i)
                target_nodes.append(j)
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(device)
    return edge_index


def get_predictions(model, dataloader: DataLoader, device: Optional[torch.device] = 'cpu'):
    model.to(device)
    model.eval()
    all_preds = []
    if isinstance(model, ShallowSGCNNet):
        adj_m, pos = adjacency_matrix_FACED()
        adj_dis_m, dm = adjacency_matrix_distance_FACED(pos, delta=6)
        edge_index = get_e_index(dm)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            if isinstance(model, ShallowSGCNNet):
                outputs = model(inputs, edge_index)
            else:
                outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().tolist())

    return all_preds


def compute_accuracies_for_all_models(
    model_root_dir: str,
    input_data: torch.Tensor,
    true_labels: torch.Tensor,
    keyfile_folder: str,
    batch_size: int = 128,
    device: Optional[torch.device] = 'cpu'
):
    dataset_size = input_data.shape[0]

    # Go through each CSV file to determine index filtering
    for file in os.listdir(keyfile_folder):
        if file.endswith(".csv") and file.startswith("Shared_Keys_"):
            model_ids = file.replace("Shared_Keys_", "").replace(".csv", "").split("_and_")
            index_path = os.path.join(keyfile_folder, file)
            indices = pd.read_csv(index_path).iloc[:, 0].values

            filtered_data = input_data[indices]
            filtered_labels = true_labels[indices]
            dataset = TensorDataset(filtered_data, filtered_labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for model_name in model_ids:
                arch_map = {
                    "Attention": "ShallowAtt",
                    "Shallow": "ShallowFBCSP",
                    "LSTM": "ShallowLSTM",
                    "RNN": "ShallowRNN",
                    "SGCN": "ShallowSGCN"
                }
                arch_folder = arch_map[model_name]
                arch_path = os.path.join(model_root_dir, arch_folder)

                if not os.path.isdir(arch_path):
                    print(f"Architecture folder not found for {model_name} ({arch_folder})")
                    continue

                print(f"Evaluating models in: {arch_folder} using indices from {file}")

                for model_file in os.listdir(arch_path):
                    if not model_file.endswith(".pth") or model_file.endswith("state.pth"):
                        continue

                    model_path = os.path.join(arch_path, model_file)
                    model = torch.load(model_path, map_location=device, weights_only=False)
                    preds = get_predictions(model, dataloader, device)

                    accuracy = np.mean(np.array(preds) == filtered_labels.numpy())
                    print(f"Accuracy for {model_file} on {file}: {accuracy:.4f}")


model_direc = "models"
keyfile_folder = "Shared_Keys"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.load("FACED_dataset/emotion_test_set.pt", map_location=device)
data = [sublist[0] for sublist in X]
data = torch.stack(data)
true_labels = torch.tensor([sublist[1] for sublist in X])

compute_accuracies_for_all_models(model_direc, data, true_labels, keyfile_folder, device=device)