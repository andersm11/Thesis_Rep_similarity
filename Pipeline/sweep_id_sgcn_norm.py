import wandb
wandb.login(key="00fce310188767f97b0cf9164a44337509fc7d54")


# === Sweep Configuration ===
sweep_config = {
    'name': 'SGCN_norm_correct_softplus sweep',
    'method': 'bayes',  # or 'random' / 'grid'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'value': 1e-3},
        'batch_size': {'values': [64]},
        'weight_decay': {'values': [0.0, 1e-4]},
        'epochs': {'value': 100},
        'kernels':{'values':[10,20,40]},
        'kernel_size':{'values':[10,25,50]},
        'pool_size':{'values':[10,20,40]},
        'num_hidden':{'values':[20,40,60]},
        'K':{'values':[1,2,4]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Master Thesis")
print(sweep_id)