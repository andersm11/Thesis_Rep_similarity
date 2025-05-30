from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import importlib
from braindecode.util import set_random_seeds
import torch.nn.functional as F
import wandb
import importlib
import os
import SGCN
from SGCN import ShallowSGCNNet
from weight_init import init_weights
from CKA_functions import adjacency_matrix_motion,adjacency_matrix_distance_motion
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
import pickle

wandb.login()
# train_set = torch.load('Datasets/emotion_train_set.pt')
# test_set = torch.load('Datasets/emotion_test_set.pt')
# train_set = torch.load('FACED_dataset/train_set.pt')
# test_set = torch.load('FACED_dataset/test_set.pt')

with open('Datasets/train_set.pkl', 'wb') as f_train:
   train_set = pickle.load(f_train)

# Save test_set to a file
with open('Datasets/test_set.pkl', 'wb') as f_test:
    test_set = pickle.load(f_test)

n_classes = 4
n_channels = 22
input_window_samples = 1125
print("n_classes: ", n_classes)
print("n_channels:", n_channels)
print("input_window_samples size:", input_window_samples)
cuda = torch.cuda.is_available()  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cuda:
    torch.backends.cudnn.benchmark = True

def train_one_epoch(
        dataloader: DataLoader, model: Module, edge_index,loss_fn,  optimizer,
        scheduler: LRScheduler, epoch: int, device, print_batch_stats=True
):
    device = next(model.parameters()).device  # Get model device
    
    model.train()  # Set the model to training mode
    train_loss, correct = 0, 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        disable=not print_batch_stats)

    for batch_idx, (X, y,_) in progress_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X,edge_index)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

    scheduler.step()

    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct

@torch.no_grad()
def test_model(dataloader: DataLoader, model: torch.nn.Module,edge_index, loss_fn, print_batch_stats=True):
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

    for batch_idx, (X, y,_) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X,edge_index)
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
    
    return test_loss, overall_accuracy, class_accuracies, all_preds, all_targets

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

def train():
    run = wandb.init(
        name="SGCN",
        tags=["Shallow", "testgroup3"]
    )
    config = wandb.config
    
    
    adj_m,pos = adjacency_matrix_motion()
    #print(adj_m)
    adj_dis_m, dm = adjacency_matrix_distance_motion(pos,delta=6)
    dm
    torch_tensor = torch.from_numpy(dm)
    edge_weight = torch_tensor.reshape(-1)
    #print(edge_weight.shape)
    model = ShallowSGCNNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        edge_weights=edge_weight,
        dropout = 0.5,
        num_kernels = config.kernels,
        kernel_size=config.kernel_size,
        pool_size=config.pool_size,
        num_hidden=config.num_hidden,
        K=config.K
    )
    edge_index = get_e_index(dm)
    if cuda:
        model.cuda()
    model.apply(init_weights)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - 1)
    loss_fn = CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)

    for epoch in range(1, config.epochs + 1):
        #print(f"Epoch {epoch}/{n_epochs}: ", end="")

        train_loss, train_accuracy = train_one_epoch(
            train_loader, model,edge_index, loss_fn, optimizer, scheduler, epoch, device
        )

        test_loss, test_accuracy, class_accuracies, batch_preds, batch_targets = test_model(test_loader, model,edge_index, loss_fn,print_batch_stats=False)
        if math.isnan(train_loss) or math.isnan(test_loss):
            wandb.log({"error": "NaN loss encountered", "epoch": epoch})
            print(f"NaN encountered in loss at epoch {epoch}, stopping early.")
            break

        final_acc = test_accuracy


        adj_matrix = model.sgconv.edge_weights.detach().cpu().numpy().reshape(22,22)

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(adj_matrix, cmap='viridis', interpolation='nearest',vmin=0)  # Adjust color map
        fig.colorbar(cax)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        pil_img = PILImage.open(buf)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy * 100,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"class_{class_idx}_accuracy": acc for class_idx, acc in class_accuracies.items()},
            "adj_matrix": wandb.Image(pil_img),  # Log the PIL image directly
        })

    wandb.finish()


wandb.agent(
    os.environ["SWEEP_ID"],
    function=train,
    project="Master Thesis",
    entity="philinthesky",  # or whatever your W&B username/team is
    count=30
)
  # Run 10 trials