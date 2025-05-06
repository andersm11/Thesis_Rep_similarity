import torch
import os
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import LSTM_model
import RNN_model
from LSTM_model import ShallowLSTMNet
from RNN_model import ShallowRNNNet
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from CKA_functions import CKA, linear_kernel, normalize_kernel, centering_matrix
from sklearn.manifold import TSNE
from itertools import combinations

def analyze_logit_similarity(logits_model1, logits_model2, preds_model1, preds_model2, output_dir="logit_analysis"):
    """
    Computes prediction agreement and various similarity metrics between logits of two models.
    Saves visualizations and metrics to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    assert logits_model1.shape == logits_model2.shape, "Logit shapes must match"
    assert preds_model1.shape == preds_model2.shape, "Prediction shapes must match"

    num_samples = logits_model1.shape[0]

    # 1. Prediction Agreement
    agreement = (preds_model1 == preds_model2).sum() / num_samples
    with open(os.path.join(output_dir, "agreement.txt"), "w") as f:
        f.write(f"Prediction agreement: {agreement:.4f}\n")
    print(f"Prediction agreement: {agreement:.4f}")

    # 2. Pearson and 3. Cosine Similarity
    pearsons, cosines, spearmans = [], [], []
    for l1, l2 in zip(logits_model1, logits_model2):
        pearsons.append(pearsonr(l1, l2)[0])
        spearmans.append(spearmanr(l1, l2).correlation)
        cosines.append(cosine_similarity(l1.reshape(1, -1), l2.reshape(1, -1))[0, 0])

    metrics = {
        "Average Pearson": np.mean(pearsons),
        "Average Cosine": np.mean(cosines),
        "Average Spearman": np.mean(spearmans),
    }

    # Save metrics
    with open(os.path.join(output_dir, "similarity_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(metrics)

    # 4. Plot Distributions of Similarity Metrics
    def plot_distribution(values, title, filename):
        plt.figure()
        plt.hist(values, bins=50, alpha=0.75, color='skyblue')
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_distribution(pearsons, "Pearson Correlation Distribution", "pearson_distribution.png")
    plot_distribution(cosines, "Cosine Similarity Distribution", "cosine_distribution.png")
    plot_distribution(spearmans, "Spearman Rank Correlation Distribution", "spearman_distribution.png")

    # 5. t-SNE and PCA on logits
    all_logits = np.vstack([logits_model1, logits_model2])
    labels = np.array(["Model 1"] * len(logits_model1) + ["Model 2"] * len(logits_model2))

    def plot_embedding(embedding, title, filename):
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(embedding[idx, 0], embedding[idx, 1], label=label, alpha=0.6)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    tsne_embedding = tsne.fit_transform(all_logits)
    plot_embedding(tsne_embedding, "t-SNE of Logits", "tsne_logits.png")

    # PCA
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(all_logits)
    plot_embedding(pca_embedding, "PCA of Logits", "pca_logits.png")

def find_base_model_files(root_dir):
    """
    Traverse the model directory tree and collect all base .pth files (not _state.pth).
    Returns a list of (model_path, model_name) tuples.
    """
    model_files = []
    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('.pth') and not fname.endswith('_state.pth'):
                full_path = os.path.join(subdir, fname)
                model_name = os.path.relpath(full_path, root_dir).replace("/", "__").replace("\\", "__")
                model_files.append((full_path, model_name))
    return model_files

def compare_all_model_logits(model_root_dir, output_dir_root="logit_similarity_outputs"):
    os.makedirs(output_dir_root, exist_ok=True)
    model_files = find_base_model_files(model_root_dir)

    for (path1, name1), (path2, name2) in combinations(model_files, 2):
        try:
            data1 = torch.load(path1, map_location="cpu")
            data2 = torch.load(path2, map_location="cpu")

            logits1, preds1 = data1["logits"], data1["preds"]
            logits2, preds2 = data2["logits"], data2["preds"]

            output_dir = os.path.join(output_dir_root, f"{name1}_vs_{name2}")
            os.makedirs(output_dir, exist_ok=True)

            analyze_logit_similarity(logits1, logits2, preds1, preds2, output_dir=output_dir)
            print(f"[✓] Compared: {name1} vs {name2}")
        except Exception as e:
            print(f"[!] Failed to compare {name1} vs {name2}: {e}")

compare_all_model_logits("cka_models")