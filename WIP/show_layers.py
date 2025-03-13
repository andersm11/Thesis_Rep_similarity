import os
import random
import torch

def load_random_model():
    # Get all files in the current directory
    files = [f for f in os.listdir() if os.path.isfile(f)]
    
    # Filter for model files (assuming they have a .pt or .pth extension)
    model_files = [f for f in files if f.endswith((".pt", ".pth"))]

    if not model_files:
        print("No model files found in the current directory.")
        return

    # Select a random model file
    random_model = random.choice(model_files)
    print(f"Loading model: {random_model}")

    # Load the model
    model = torch.load(random_model, map_location=torch.device("cpu"))

    # Print all layer names
    print("\nModel Layers:")
    if isinstance(model, torch.nn.Module):
        for name, layer in model.named_modules():
            print(name, "->", layer)
    else:
        print("Loaded object is not a PyTorch model.")

load_random_model()
