import numpy as np
import os
import torch
import json
# from Spatial_attention_FACED import ShallowAttentionNet
# from SGCN_FACED import ShallowSGCNNet
# from shallow_laurits_faced import ShallowFBCSPNet
# from RNN_model import ShallowRNNNet
# from LSTM_model import ShallowLSTMNet
import pickle
from typing import Optional
model_direc = "../motion_models"

def get_predictions(model, input_data: torch.Tensor, device: Optional[torch.device] = 'cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_data)
        predicted = torch.argmax(outputs, dim=1)
    return predicted.cpu().tolist()

def save_predictions_for_all_models(
    model_root_dir: str,
    input_data: torch.Tensor,
    true_labels: torch.Tensor,
    output_dir: str,
    device: Optional[torch.device] = 'cpu'
):
    os.makedirs(output_dir, exist_ok=True)

    # Save the ground truth labels
    targets_path = os.path.join(output_dir, "targets.json")
    true_labels = true_labels.cpu().tolist()
    with open(targets_path, "w") as f:
        json.dump(true_labels, f, indent=4)

    print(f"Saved targets to {targets_path} with {len(true_labels)} labels.")

    # Go through each architecture subfolder
    for arch in os.listdir(model_root_dir):
        arch_path = os.path.join(model_root_dir, arch)
        if not os.path.isdir(arch_path):
            continue

        arch_output_path = os.path.join(output_dir, arch)
        os.makedirs(arch_output_path, exist_ok=True)

        for model_file in os.listdir(arch_path):
            if not model_file.endswith(".pth"):
                continue

            model_path = os.path.join(arch_path, model_file)
            print(f"Processing model: {model_path}", flush=True)

            model = torch.load(model_path, map_location=device,weights_only=False)
            preds = get_predictions(model, input_data, device)

            # Sanity check
            if len(preds) != len(true_labels):
                raise ValueError(f"Prediction length mismatch for {model_file}: got {len(preds)} vs {len(true_labels)}")

            pred_file = os.path.splitext(model_file)[0] + "_preds.json"
            pred_path = os.path.join(arch_output_path, pred_file)
            with open(pred_path, "w") as f:
                json.dump(preds, f, indent=4)

            print(f"Saved predictions to {pred_path}")

    print("✅ All predictions and targets saved successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('../Datasets/test_set.pkl','rb') as f:
    X = pickle.load(f)
def to_tensor(dataset):
        X_list, y_list = [], []
        for x, y,_ in dataset:
            X_list.append(torch.tensor(x, dtype=torch.float32))
            y_list.append(torch.tensor(y, dtype=torch.long))
        return torch.stack(X_list), torch.stack(y_list)

X_, y_ = to_tensor(X)
true_labels = y_  # Extract the second element (labels)
print("data shape:", X_.shape)
print("labels shape:", len(true_labels))
save_predictions_for_all_models(model_direc,X_,true_labels,"predictions",device=device)