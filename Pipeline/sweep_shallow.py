from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import importlib
from braindecode.util import set_random_seeds
import torch.nn.functional as F
import wandb
import importlib
import os
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
def train():
    run = wandb.init()
    config = wandb.config

    model = ShallowFBCSPNet(
        n_chans=eeg.shape[1],
        n_subjs=9,
        n_outputs=3,
        n_times=eeg.shape[2],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - 1)
    loss_fn = CrossEntropyLoss()

    train_loader = DataLoader(train_tensor, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=config.batch_size)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, device)
        test_loss, test_accuracy, class_accuracies, batch_preds, batch_targets = test_model(test_loader, model, loss_fn)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy * 100,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            **{f"class_{class_idx}_acc": acc for class_idx, acc in class_accuracies.items()}
        })

    wandb.finish()

# === Sweep Configuration ===
sweep_config = {
    'method': 'bayes',  # or 'random' / 'grid'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [1e-3, 5e-4, 1e-4]},
        'batch_size': {'values': [64, 124, 256]},
        'weight_decay': {'values': [0.0, 1e-4, 1e-2]},
        'epochs': {'value': 100}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Master Thesis")
wandb.agent(sweep_id, function=train, count=10)  # Run 10 trials