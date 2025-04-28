import wandb
wandb.login(key="00fce310188767f97b0cf9164a44337509fc7d54")


# === Sweep Configuration ===
sweep_config = {
    'name': 'SGCN_no_norm sweep',
    'method': 'bayes',  # or 'random' / 'grid'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'lr': {'value': 1e-3},
        'batch_size': {'values': [32,64,128]},
        'weight_decay': {'values': [0.0, 1e-4]},
        'epochs': {'value': 100},
        'kernels':{'values':[10,20]},
        'kernel_size':{'values':[25,50,80]},
        'pool_size':{'values':[20,40,70]},
        'num_hidden':{'values':[40,60,80]},
        'K':{'values':[1,2,4]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Master Thesis")
print(sweep_id)