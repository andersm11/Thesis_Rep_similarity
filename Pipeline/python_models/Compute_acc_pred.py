import os
import json
import numpy as np
import pandas as pd

def compute_accuracy_from_saved_preds(predictions_dir: str, keyfile_dir: str):
    # Load true labels
    with open(os.path.join(predictions_dir, "targets.json")) as f:
        true_labels = json.load(f)
    print(f"Loaded {len(true_labels)} true labels (unfiltered)")

    for keyfile in os.listdir(keyfile_dir):
        if not (keyfile.endswith(".csv") and keyfile.startswith("Shared_Keys_")):
            continue

        # Parse model names from filename
        model_names = keyfile.replace("Shared_Keys_", "").replace(".csv", "").split("_and_")
        indices = pd.read_csv(os.path.join(keyfile_dir, keyfile)).iloc[:, 0].tolist()

        # Subset true labels by indices
        filtered_labels = np.array(true_labels)[indices]
        print(f"\n📂 Keyfile: {keyfile}")
        print(f"Filtered label count: {len(filtered_labels)}")

        for model_name in model_names:
            model_subdir = model_name.replace("Shallow", "ShallowFBCSP") if model_name == "Shallow" else f"Shallow{model_name}"
            arch_path = os.path.join(predictions_dir, model_subdir)

            if not os.path.exists(arch_path):
                print(f"Skipping: no predictions folder for {model_subdir}")
                continue

            print(f"Evaluating models in: {model_subdir}")

            for file in os.listdir(arch_path):
                if not file.endswith("_preds.json"):
                    continue

                with open(os.path.join(arch_path, file)) as f:
                    preds = json.load(f)
                print(f"  ↳ {file}: {len(preds)} predictions loaded (unfiltered)")

                filtered_preds = np.array(preds)[indices]
                print(f"    Filtered predictions: {len(filtered_preds)}")

                if len(filtered_preds) != len(filtered_labels):
                    print(f"    ❌ Length mismatch in {file} for {keyfile}")
                    continue

                acc = np.mean(filtered_preds == filtered_labels)
                print(f"    ✅ Accuracy: {acc:.4f}")

# Example usage
compute_accuracy_from_saved_preds(predictions_dir="predictions", keyfile_dir="Shared_Keys")
