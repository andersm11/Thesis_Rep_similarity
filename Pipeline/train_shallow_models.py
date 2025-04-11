import importlib
from braindecode.util import set_random_seeds
import torch.nn.functional as F
import wandb
import importlib
import shallow_laurits_faced
from shallow_laurits_faced import ShallowFBCSPNet
from weight_init import init_weights
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image as PILImage
import io


wandb.login()
# train_set = torch.load('Datasets/emotion_train_set.pt')
# test_set = torch.load('Datasets/emotion_test_set.pt')
train_set = torch.load('FACED_dataset/emotion_train_set.pt')
test_set = torch.load('FACED_dataset/emotion_test_set.pt')


cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 282828
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 3
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_channels = 32
input_window_samples = 400

print("n_classes: ", n_classes)
print("n_channels:", n_channels)
print("input_window_samples size:", input_window_samples)


# Define a method for training one epoch


def train_one_epoch(
        dataloader: DataLoader, model: Module, loss_fn, optimizer,
        scheduler: LRScheduler, epoch: int, device, print_batch_stats=True
):
    model.train()  # Set the model to training mode
    train_loss, correct = 0, 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        disable=not print_batch_stats)

    for batch_idx, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        #print(y)
        #print(X.shape)
        optimizer.zero_grad()
        pred = model(X)


        #print(y.shape)
        #print(pred.shape)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()
        #print(loss.item())
        #print(loss)
        #print(train_loss)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        #if print_batch_stats:
        #    progress_bar.set_description(
        #        f"Epoch {epoch}/{n_epochs}, "
        #        f"Batch {batch_idx + 1}/{len(dataloader)}, "
        #        f"Loss: {loss.item():.6f}"
        #    )

    # Update the learning rate
    scheduler.step()

    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


@torch.no_grad()
def test_model(dataloader: DataLoader, model: torch.nn.Module, loss_fn, print_batch_stats=True):
    device = next(model.parameters()).device  # Get model device
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()  # Switch to evaluation mode
    test_loss, correct = 0, 0

    # Initialize dictionaries for per-class tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Lists to store true and predicted labels for confusion matrix
    all_preds = []
    all_targets = []

    if print_batch_stats:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)

    for batch_idx, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        batch_loss = loss_fn(pred, y).item()

        test_loss += batch_loss
        correct += (pred.argmax(1) == y).sum().item()

        # Store predictions and true labels for confusion matrix
        all_preds.append(pred.argmax(1).cpu())
        all_targets.append(y.cpu())

        # Compute per-class accuracy
        preds_labels = pred.argmax(1)
        for label, pred_label in zip(y, preds_labels):
            class_total[label.item()] += 1
            class_correct[label.item()] += (label == pred_label).item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {batch_loss:.6f}"
            )

    # Convert lists to tensors
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute per-class accuracy
    class_accuracies = {
        cls: (class_correct[cls] / class_total[cls]) * 100 if class_total[cls] > 0 else 0
        for cls in class_total
    }

    # Compute overall accuracy
    test_loss /= n_batches
    overall_accuracy = (correct / size) * 100

    # # Print per-class accuracy
    # print("\nClass-wise Accuracy:")
    # for cls, acc in class_accuracies.items():
    #     print(f"  Class {cls}: {acc:.2f}%")

    # print(f"Test Accuracy: {overall_accuracy:.1f}%, Test Loss: {test_loss:.6f}\n")



    return test_loss, overall_accuracy, class_accuracies, all_preds, all_targets


save_path ="models/"

seeds = [99999]#,182726,91111222,44552222,12223111,100300,47456655,4788347,77766666,809890]
for _seed in seeds:
    seed = _seed
    set_random_seeds(seed=seed, cuda=cuda)
    #print(edge_weight.shape)
    model = ShallowFBCSPNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
    )

    if cuda:
        model.cuda()

    # Initialize Weights & Biases
    wandb.init(project="Master Thesis", name=f"{model.__class__.__name__} {seed}")
    model.apply(init_weights)
    # Define hyperparameters
    lr = 1e-5
    weight_decay = 1e-4
    batch_size = 64  # Start with 124
    n_epochs = 5000

    final_acc = 0.0

    # Log hyperparameters to wandb
    wandb.config.update({
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": n_epochs
    })
    # Create optimizer with per-parameter learning rates
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - 1)

    # Define loss function
    loss_fn = CrossEntropyLoss()

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Initialize lists to store all predictions & targets
    # all_preds, all_targets = [], []

    # Training loop
    for epoch in range(1, n_epochs + 1):
        #print(f"Epoch {epoch}/{n_epochs}: ", end="")

        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, device
        )

        test_loss, test_accuracy, class_accuracies, batch_preds, batch_targets = test_model(test_loader, model, loss_fn,print_batch_stats=False)
        final_acc = test_accuracy

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy * 100,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"class_{class_idx}_accuracy": acc for class_idx, acc in class_accuracies.items()},
        })


    # all_preds = np.array(all_preds)
    # all_targets = np.array(all_targets)

    # # Save predictions & true labels for later use (confusion matrix)
    # wandb.log({"all_preds": all_preds.tolist(), "all_targets": all_targets.tolist()})
    wandb.finish()

    torch.save(model, save_path+f"{model.__class__.__name__}_{math.ceil(final_acc)}_{seed}.pth")
    torch.save(model.state_dict(), save_path+f"{model.__class__.__name__}_{math.ceil(final_acc)}_{seed}_state.pth")



