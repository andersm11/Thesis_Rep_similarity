from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def get_labels(model,dataloader):
    model.eval()
    data = enumerate(dataloader)
    all_preds = []
    all_truths = []
    for batch_idx, (X, y, _) in data:
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
    print(pred)
    print(truth)
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
    sns.heatmap(cm_percentage, annot=annotations, fmt="", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Raw Count and Percentage)")
    plt.show()
