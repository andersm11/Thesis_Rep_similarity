import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cka_heatmap_with_acc(cka_matrix, model_names, model_accuracies, output_folder, title="CKA Heatmap"):
    """
    Generate a 3x3 CKA heatmap with accuracy annotations.

    Args:
        cka_matrix (np.ndarray): A 3x3 numpy array of CKA values.
        model_names (list): List of 3 model name strings.
        model_accuracies (list): List of 3 corresponding accuracy values (floats or ints).
        output_folder (str): Path to save the heatmap image.
        title (str): Title of the heatmap.
    """
    assert cka_matrix.shape == (3, 3), "cka_matrix must be 3x3"
    assert len(model_names) == len(model_accuracies) == 3, "Need exactly 3 model names and accuracies"

    os.makedirs(output_folder, exist_ok=True)

    # Prepare model labels with accuracies
    model_acc_map = {name: acc for name, acc in zip(model_names, model_accuracies)}
    x_labels = [f"{name}\n({acc:.0f}%)" for name, acc in zip(model_names, model_accuracies)]
    #y_labels = x_labels
    y_labels = list(reversed(x_labels))  # For flipped matrix display

    # Create annotation matrix (CKA + acc info)
    annotations = [["" for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            cka_val = cka_matrix[i, j]
            acc_i = model_accuracies[i]
            acc_j = model_accuracies[j]
            annotations[i][j] = f"{cka_val:.2f}\n({acc_i:.0f}% vs {acc_j:.0f}%)"

    # Flip matrix for visual alignment
    #matrix_vals = np.flipud(cka_matrix)
    #annotations = annotations[::-1]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cka_matrix,
        annot=annotations,
        fmt='',
        cmap='gist_heat',
        square=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.5,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={"size": 18, "weight": "bold"}
    )
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    plt.title(title, fontsize=22)
    plt.xlabel("Model (Accuracy)", fontsize=18)
    plt.ylabel("Model (Accuracy)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    filepath = os.path.join(output_folder, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Heatmap saved to {filepath}")
    
cka_matrix = np.array([
    [0.67, 0.85, 0.89],
    [0.67, 0.90, 0.85],
    [0.99, 0.67, 0.67]
])
model_names = ["ShallowFBCSP", "ShallowLSTM", "ShallowRNN"]
accuracies = [61, 59, 58]

plot_cka_heatmap_with_acc(cka_matrix, model_names, accuracies, output_folder="motion_cka",title="CKA Extraction Layers Temporal (Motion)")