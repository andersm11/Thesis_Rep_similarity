from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
import scipy.stats as stats 
import pandas as pd
import itertools
import math
from torch.utils.data import DataLoader
from CKA_functions import load_dataset, load_model,adjacency_matrix_motion,adjacency_matrix_distance_motion
from SGCN import ShallowSGCNNet
CLASS_LABELS = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]

def get_labels(model,dataloader):
    model.eval()
    data = enumerate(dataloader)
    all_preds = []
    all_truths = []
    if isinstance(model, ShallowSGCNNet):
        adj_m,pos = adjacency_matrix_motion()
        adj_dis_m, dm = adjacency_matrix_distance_motion(pos,delta=5)
        threshold = 0  # Adjust as needed
        source_nodes = []
        target_nodes = []

        # Iterate over all elements in the distance matrix, including self-loops and duplicates
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):  # Iterate over all pairs, including (i, i)
                if dm[i, j] >= threshold:  # If the distance meets the condition
                    source_nodes.append(i)  # Source node
                    target_nodes.append(j)  # Target node
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    try:
        for batch_idx, (X, y, _) in data:
            if isinstance(model,ShallowSGCNNet):
                preds = model(X,edge_index)
            else:
                preds = model(X)
            all_truths.append(y)
            pred_labels = torch.argmax(preds, dim=1)  # Use torch.argmax to get the predicted labels
            all_preds.append(pred_labels)  # Append the tensor (not a list)
    except Exception as e:
        print(e)
        print("trying to 2 outputs only")
        for batch_idx, (X, y) in data:
            if isinstance(model,ShallowSGCNNet):
                preds = model(X,edge_index)
            else:
                preds = model(X)
            all_truths.append(y)
            pred_labels = torch.argmax(preds, dim=1)  # Use torch.argmax to get the predicted labels
            all_preds.append(pred_labels)  # Append the tensor (not a list)

    # Concatenate all the predictions into a single tensor
    all_preds = torch.cat(all_preds, dim=0)  # Concatenate the list of tensors
    all_truths = torch.cat(all_truths,dim=0)
    return all_preds,all_truths

def compute_accuracy(pred,truth):
    assert len(pred) == len(truth)
    n = len(pred)
    correct = 0
    for p,t in zip(pred,truth):
        if p == t:
            correct +=1
    return correct/n

def compute_class_confusions(pred, truth):
    cm = confusion_matrix(truth, pred)

    # Calculate percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Row-wise normalization to get percentages

    # Define class labels (modify if needed)
    class_labels = [f"Class {i}" for i in range(cm.shape[0])]

    # Create the annotations with both raw count and percentage
    annotations = [
        [f"{cm[i, j]} ({cm_percentage[i, j]:.2f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ]

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=annotations, fmt="", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Raw Count and Percentage)")
    plt.show()
    
def compute_model_confusion(model, X_test, y_true):
    """Compute confusion matrix for a given model."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    return confusion_matrix(y_true, preds)

def plot_and_save_confusion_matrix(cm_avg, CLASS_LABELS, output_path, model_name):
    """Plot and save averaged confusion matrix with colors based on percentage values."""
    
    # Compute percentage-based confusion matrix
    cm_percentage = cm_avg.astype('float') / cm_avg.sum(axis=1, keepdims=True) * 100
    
    # Create annotations with raw count and percentage
    annotations = [[f"{cm_avg[i, j]} ({cm_percentage[i, j]:.2f}%)" 
                    for j in range(cm_avg.shape[1])] 
                   for i in range(cm_avg.shape[0])]

    # Plot with color scaling from 0% to 100%
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=annotations, fmt="", cmap="Blues", xticklabels=CLASS_LABELS, 
                yticklabels=CLASS_LABELS, vmin=0, vmax=100, linewidths=0.5, linecolor='gray')

    # Labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {model_name}")

    # Save the figure
    plt.savefig(os.path.join(output_path, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()


def compute_all_model_confusion(models_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    data_path = "../Datasets/"
    X = load_dataset("test_set.pkl", data_path)
    data_loader = DataLoader(X, batch_size=16)
    
    models_cm_avg = {}  # Store average confusion matrices for all models
    
    for arch_folder in os.listdir(models_path):
        arch_path = os.path.join(models_path, arch_folder)
        if not os.path.isdir(arch_path):
            continue
        
        cm_list = []
        model_count = 0
        
        for model_file in os.listdir(arch_path):
            if not(model_file.endswith("state.pth")) and model_file.endswith(".pth"):  # Ensure only full model files are used
                model = load_model(model_file, arch_path)  # Load full model
                model_preds, truths = get_labels(model, dataloader=data_loader)
                cm = confusion_matrix(truths, model_preds)  # Fix argument order
                cm_list.append(cm)
                model_count += 1
        
        if model_count > 0:
            cm_avg = np.floor(np.mean(cm_list, axis=0)).astype(int)
            #CLASS_LABELS = [f"Class {i}" for i in range(cm_avg.shape[0])]
            save_classification_report(truths,model_preds,CLASS_LABELS,output_path,f"{model.__class__.__name__}_report.txt")
            
            # Store averaged confusion matrix for later comparison
            models_cm_avg[arch_folder] = cm_avg  
            
            # Save individual confusion matrix
            plot_and_save_confusion_matrix(cm_avg, CLASS_LABELS, output_path, arch_folder)
    
    # Generate and save class-wise accuracy comparison plot
    if models_cm_avg:
        plot_and_save_class_accuracy(models_cm_avg, CLASS_LABELS, output_path)
        plot_and_save_overall_accuracy(models_cm_avg,output_path)
        plot_and_save_mislabeling_histograms(models_cm_avg,CLASS_LABELS,output_path)
        plot_and_save_predicted_label_distributions(models_cm_avg,CLASS_LABELS,output_path)
        plot_and_save_overall_prediction_distributions(models_cm_avg,CLASS_LABELS,output_path)
        f1_scores = compute_f1_scores(models_cm_avg,CLASS_LABELS)
        plot_and_save_f1_scores(f1_scores,output_path)
        save_epi_test(models_cm_avg,output_path)
            
def plot_and_save_class_accuracy(models_cm_avg, class_labels, output_path):
    """Plot and save histogram comparing accuracy per class across multiple models."""
    
    # Create a DataFrame to store class accuracy for all models
    data = []
    for model_name, cm_avg in models_cm_avg.items():
        class_accuracy = np.diag(cm_avg) / cm_avg.sum(axis=1) * 100
        for label, acc in zip(class_labels, class_accuracy):
            data.append({"Model": model_name, "Class": label, "Accuracy": acc})
    
    df = pd.DataFrame(data)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Class", y="Accuracy", hue="Model", data=df)
    
    # Labels and title
    plt.xlabel("Class Label")
    plt.ylabel("Accuracy (%)")
    plt.title("Class-wise Accuracy Comparison Across Models")
    plt.ylim(0, 100)
    
    # Save the figure
    plt.savefig(os.path.join(output_path, "class_accuracy_comparison.png"), dpi=300)
    plt.close()
    
def plot_and_save_overall_accuracy(models_cm_avg, output_path):
    """Plot and save overall accuracy comparison between models."""
    
    # Calculate overall accuracy for each model (sum of diagonal elements / sum of all elements)
    overall_accuracy = {model_name: np.diag(cm_avg).sum() / cm_avg.sum() * 100 for model_name, cm_avg in models_cm_avg.items()}
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(list(overall_accuracy.items()), columns=["Model", "Accuracy"])
    
    # Plot overall accuracy comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Accuracy", data=df, hue="Model")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black')
    
    # Labels and title
    plt.xlabel("Model")
    plt.ylabel("Overall Accuracy (%)")
    plt.title("Overall Accuracy Comparison Between Models")
    plt.ylim(0, 100)
    
    # Save the figure
    plt.savefig(os.path.join(output_path, "overall_accuracy_comparison.png"), dpi=300)
    plt.close()
    
def plot_and_save_mislabeling_histograms(models_cm_avg, class_labels, output_path):
    """Plot and save barplots showing how each model mislabels a specific class."""
    
    # Create a directory to save individual class mislabeling histograms
    os.makedirs(os.path.join(output_path, "mislabeling_histograms"), exist_ok=True)
    
    # Iterate through each class and create barplots for mislabeling across models
    for class_index, class_label in enumerate(class_labels):
        mislabel_data = []

        # Collect mislabel data for the current class across all models
        for model_name, cm_avg in models_cm_avg.items():
            # Get the row corresponding to the current class (i.e., predictions for this class)
            mislabels = cm_avg[class_index, :].copy()  # All predictions for the current class
            
            # Remove the diagonal (correct classifications)
            mislabels[class_index] = 0
            
            # For each mislabel, store the misclassified label
            for other_class, count in enumerate(mislabels):
                if count > 0:
                    mislabel_data.extend([{
                        "Misclassified As": class_labels[other_class], 
                        "Count": count,
                        "Model": model_name
                    }] * count)  # Repeat the mislabel count for each misclassification

        # Create a DataFrame for plotting mislabel data
        mislabel_df = pd.DataFrame(mislabel_data, columns=["Model", "Misclassified As", "Count"])

        # Plot the barplot of mislabeling for this class
        plt.figure(figsize=(10, 6))
        sns.barplot(data=mislabel_df, x="Misclassified As", y="Count", hue="Model", 
                    errorbar=None)

        # Labels and title
        plt.xlabel("Misclassified As")
        plt.ylabel("Mislabel Count")
        plt.title(f"Mislabeling of Class: {class_label}")

        # Save the plot for this class
        plt.savefig(os.path.join(output_path, "mislabeling_histograms", f"{class_label}_mislabeling_histogram.png"), dpi=300)
        plt.close()
        
def plot_and_save_predicted_label_distributions(models_cm_avg, class_labels, output_path):
    """Plot and save histograms of predicted label distributions for each class."""
    
    # Create a directory to save the predicted label distributions
    os.makedirs(os.path.join(output_path, "predicted_label_distributions"), exist_ok=True)
    
    # Iterate through each class and create histograms of predicted label distributions
    for class_index, class_label in enumerate(class_labels):
        predicted_data = []

        # Iterate through all models and capture the predicted labels for the current class
        for model_name, cm_avg in models_cm_avg.items():
            # Get the column corresponding to the current class, which represents the predictions for this class
            predicted_counts = cm_avg[:, class_index]  # All predictions for the current class
            
            # Add the predicted counts for this model to the predicted_data list
            predicted_data.extend([(model_name, predicted_label, count) 
                                   for predicted_label, count in enumerate(predicted_counts) if count > 0])

        # Create a DataFrame for plotting
        predicted_df = pd.DataFrame(predicted_data, columns=["Model", "Predicted As", "Count"])

        # Plot the histogram of predicted label distributions for this class
        plt.figure(figsize=(10, 6))
        sns.barplot(data=predicted_df, x="Predicted As", y="Count", hue="Model", 
                    errorbar=None)
        
        # Labels and title
        plt.xlabel("Predicted Label")
        plt.ylabel("Count")
        plt.title(f"Predicted Label Distribution for Class: {class_label}")
        
        # Save the plot
        plt.savefig(os.path.join(output_path, "predicted_label_distributions", f"{class_label}_predicted_distribution.png"), dpi=300)
        plt.close()
        
def compute_f1_scores(models_cm_avg, class_labels):
    """
    Computes the F1-score for each model based on the confusion matrix average.
    
    Args:
    - models_cm_avg (dict): A dictionary of model names as keys and confusion matrices as values.
    - class_labels (list): List of class labels (task names).
    
    Returns:
    - f1_scores (dict): A dictionary of model names as keys and their F1-scores as values.
    """
    f1_scores = {}
    
    for model_name, cm_avg in models_cm_avg.items():
        true_positives = np.diag(cm_avg)  # Diagonal elements represent the true positives
        false_positives = cm_avg.sum(axis=0) - true_positives  # Sum of columns minus diagonal gives false positives
        false_negatives = cm_avg.sum(axis=1) - true_positives  # Sum of rows minus diagonal gives false negatives
        
        # Handle the case where there are no true positives, false positives, or false negatives
        precision = true_positives / (true_positives + false_positives + 1e-10)  # Prevent division by zero
        recall = true_positives / (true_positives + false_negatives + 1e-10)  # Prevent division by zero
        
        # F1-Score for each class
        f1_class = 2 * (precision * recall) / (precision + recall + 1e-10)
        # Compute average F1-Score (macro average)
        avg_f1 = np.mean(f1_class)
        f1_scores[model_name] = avg_f1
    
    return f1_scores

def plot_and_save_f1_scores(f1_scores, output_path):
    """
    Plots and saves the F1-score comparison across models with value labels.
    
    Args:
    - f1_scores (dict): A dictionary of model names and their corresponding F1-scores.
    - output_path (str): The path where the F1-score comparison plot will be saved.
    """
    # Convert f1_scores to a DataFrame for plotting
    print(f1_scores)
    f1_df = pd.DataFrame(list(f1_scores.items()), columns=["Model", "F1-Score"])
    
    # Plot the F1-scores for each model
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="F1-Score", data=f1_df, hue="Model", edgecolor="black")
    
    # Add value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black')
    
    # Labels and title
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title("F1-Score Comparison Between Models", fontsize=14)
    plt.ylim(0, 1)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha='right')

    # Save the figure
    plt.savefig(os.path.join(output_path, "f1_scores_comparisons.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_classification_report(truths, model_preds, class_labels, output_path, filename="classification_report.txt"):
    """Generate and save the classification report to a text file."""
    
    # Generate the classification report
    report = classification_report(truths, model_preds, target_names=class_labels)
    
    # Specify the output file path
    report_file_path = os.path.join(output_path, filename)
    
    # Write the classification report to a file
    with open(report_file_path, "w") as file:
        file.write(report)
        
    print(f"Classification report saved to: {report_file_path}")

def save_epi_test(all_cms,output_path):
    def epi_tests_ova(cm, target_class, class_labels):
        idx = class_labels.index(target_class)
        TP = cm[idx, idx]
        FP = sum(cm[idx,:]) - TP
        FN = sum(cm[:, idx]) - TP
        TN = cm.sum() - (TP + FP + FN)
        
        contingency_table = np.array([[TP, FP], [FN, TN]])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)  # Fixed function call
        
        return {"Class": target_class, "TP": TP, "FP": FP, "FN": FN, "TN": TN, "Chi2": chi2, "p-value": p}
    
    for model_name, cm_avg in all_cms.items():

        results = [epi_tests_ova(cm_avg, cls, CLASS_LABELS) for cls in CLASS_LABELS]
        summary_table = pd.DataFrame(results)
        summary_table.iloc[:, 1:] = summary_table.iloc[:, 1:].apply(lambda x: np.round(x, 2))
        print("\nSummary Table of Metrics:")
        print(summary_table)
        summary_table.to_csv(f"{output_path}/{model_name}_multi_class_metrics.csv", index=False)

def plot_and_save_overall_prediction_distributions(models_cm_avg, class_labels, output_path):
    """Plot and save overall histograms of predicted label distributions for each model."""
    
    # Create a directory to save the overall predicted label distributions
    os.makedirs(os.path.join(output_path, "overall_prediction_distributions"), exist_ok=True)
    
    overall_data = []
    
    # Iterate through each model and capture overall predicted label distributions
    for model_name, cm_avg in models_cm_avg.items():
        # Sum across rows to get total predictions per class
        total_predictions = cm_avg.sum(axis=0)
        
        # Store the predicted counts for each class
        overall_data.extend([(model_name, class_labels[class_idx], count) 
                             for class_idx, count in enumerate(total_predictions) if count > 0])
    
    # Create a DataFrame for plotting
    overall_df = pd.DataFrame(overall_data, columns=["Model", "Predicted Label", "Count"])
    
    # Plot the histogram of overall predicted label distributions
    plt.figure(figsize=(10, 6))
    sns.barplot(data=overall_df, x="Predicted Label", y="Count", hue="Model", errorbar=None)
    
    # Labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("Count")
    plt.title("Overall Predicted Label Distribution Across Models")
    
    # Save the plot
    plt.savefig(os.path.join(output_path, "overall_prediction_distributions", "overall_predicted_distribution.png"), dpi=300)
    plt.close()
    
