import wandb
wandb.login(key="00fce310188767f97b0cf9164a44337509fc7d54")


# === Sweep Configuration ===
sweep_config = {
    'name': 'Shallow - Spatial sweep',
    'method': 'bayes',  # or 'random' / 'grid'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'value': 1e-3},
        'batch_size': {'values': [32,64, 128]},
        'weight_decay': {'values': [0.0, 1e-4]},
        'dropout': {'values':[0.5,0.7]},
        'epochs': {'value': 100},
        'kernels':{'values':[40,50,60]},
        'kernel_size':{'values':[25,50,80]},
        'pool_size':{'values':[25,50,100,200]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Master Thesis")
print(sweep_id)