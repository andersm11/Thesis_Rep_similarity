import wandb
wandb.login(key="00fce310188767f97b0cf9164a44337509fc7d54")

# === Sweep Configuration ===
sweep_config = {
    'name': 'Spatial Att overtime sweep',
    'method': 'bayes',  # or 'random' / 'grid'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'value': 1e-3},
        'batch_size': {'values': [32,64, 256]},
        'weight_decay': {'values': [0.0, 1e-2]},
        'epochs': {'value': 100},
        'kernels':{'values':[40,60,80]},
        'kernel_size':{'values':[25,50]},
        'pool_size':{'values':[30,50,80,100]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Master Thesis")
print(sweep_id)