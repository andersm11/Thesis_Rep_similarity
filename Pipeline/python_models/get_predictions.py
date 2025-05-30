import numpy as np
import os
import torch
import json
from Spatial_attention_FACED import ShallowAttentionNet
from SGCN_FACED_norm import ShallowSGCNNet
from shallow_laurits_faced import ShallowFBCSPNet
from RNN_model import ShallowRNNNet
from LSTM_model import ShallowLSTMNet
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from CKA_functions import adjacency_matrix_FACED, adjacency_matrix_distance_FACED


def get_e_index(dm):
  threshold = 0  # Adjust as needed

  source_nodes = []
  target_nodes = []

  # Iterate over all elements in the distance matrix, including self-loops and duplicates
  for i in range(dm.shape[0]):
      for j in range(dm.shape[1]):  # Iterate over all pairs, including (i, i)
          if dm[i, j] >= threshold:  # If the distance meets the condition
              source_nodes.append(i)  # Source node
              target_nodes.append(j)  # Target node

  # Create the edge_index tensor
  edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(device)
  return edge_index


def get_predictions(model, dataloader: DataLoader, device: Optional[torch.device] = 'cpu'):
    model.to(device)
    model.eval()
    all_preds = []
    if isinstance(model, ShallowSGCNNet):
        adj_m,pos = adjacency_matrix_FACED()
        adj_dis_m, dm = adjacency_matrix_distance_FACED(pos,delta=6)
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

def save_predictions_for_all_models(
    model_root_dir: str,
    input_data: torch.Tensor,
    true_labels: torch.Tensor,
    output_dir: str,
    batch_size: int = 128,
    device: Optional[torch.device] = 'cpu'
):
    os.makedirs(output_dir, exist_ok=True)

    # Save the ground truth labels
    targets_path = os.path.join(output_dir, "targets.json")
    true_labels_list = true_labels.cpu().tolist()
    with open(targets_path, "w") as f:
        json.dump(true_labels_list, f, indent=4)
    print(f"Saved targets to {targets_path} with {len(true_labels_list)} labels.")

    dataset = TensorDataset(input_data, true_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Go through each architecture subfolder
    for arch in os.listdir(model_root_dir):
        arch_path = os.path.join(model_root_dir, arch)
        if not os.path.isdir(arch_path):
            continue

        arch_output_path = os.path.join(output_dir, arch)
        os.makedirs(arch_output_path, exist_ok=True)

        for model_file in os.listdir(arch_path):
            if not model_file.endswith(".pth") or model_file.endswith("state.pth"):
                continue

            model_path = os.path.join(arch_path, model_file)
            print(f"Processing model: {model_path}", flush=True)

            model = torch.load(model_path, map_location=device, weights_only=False)
            preds = get_predictions(model, dataloader, device)

            # Sanity check
            if len(preds) != len(true_labels_list):
                raise ValueError(f"Prediction length mismatch for {model_file}: got {len(preds)} vs {len(true_labels_list)}")

            pred_file = os.path.splitext(model_file)[0] + "_preds.json"
            pred_path = os.path.join(arch_output_path, pred_file)
            with open(pred_path, "w") as f:
                json.dump(preds, f, indent=4)

            print(f"Saved predictions to {pred_path}")

    print("✅ All predictions and targets saved successfully.")
model_direc = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load("FACED_dataset/emotion_test_set.pt",map_location=device)
data = [sublist[0] for sublist in X]  # Extract the first tensor in each tuple
data = torch.stack(data)  # Stack the tensors into a single tensor
true_labels = [sublist[1] for sublist in X]  # Extract the second element (labels)
true_labels = torch.tensor(true_labels)  # Convert to tensor
print("data shape:", data.shape)
print("labels shape:", len(true_labels))
save_predictions_for_all_models(model_direc,data,true_labels,"predictions",device=device)