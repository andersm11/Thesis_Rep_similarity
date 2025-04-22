import wandb
wandb.login(key="00fce310188767f97b0cf9164a44337509fc7d54")


# === Sweep Configuration ===
# === Sweep Configuration ===
sweep_config = {
    'name': 'SGCN sweep',
    'method': 'bayes',  # or 'random' / 'grid'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [1e-3, 1e-4, 5e-5]},
        'batch_size': {'values': [32,64, 128, 256]},
        'weight_decay': {'values': [0.0, 1e-4, 1e-2]},
        'epochs': {'value': 100},
        'kernels':{'values':[5,10,20,40]},
        'kernel_size':{'values':[10,25,50]},
        'pool_size':{'values':[10,20,30]},
        'num_hidden':{'values':[10,20,40]},
        'K':{'values':[1,2,3,4]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Master Thesis")
print(sweep_id)