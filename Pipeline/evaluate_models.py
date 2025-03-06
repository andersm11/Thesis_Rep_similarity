import os
import torch
from torch.utils.data import DataLoader
from CKA_functions import load_dataset, load_model
import pickle

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    #print("in acc")
    with torch.no_grad():  # Disable gradient calculations
        #print("with")
        for inputs, labels in test_loader:
            #print("test and labels")
            inputs, labels = inputs.to(device), labels.to(device)
            #print("got input and labels")
            
            outputs = model(inputs)
            #print("got outputs")
            predictions = torch.argmax(outputs, dim=1)
            #print(predictions)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy

# Function to load and evaluate all models in a directory
def evaluate_all_models(models_dir, test_loader, device):
    results = {}

    # List all files in the directory
    for filename in os.listdir(models_dir):
        if filename.endswith(".pt") or filename.endswith(".pth"):  # Check for model files
            model_path = os.path.join(models_dir, filename)
            #print(model_path)
            try:
                # Load the full model
                model = load_model(filename,"models")
                #print(model)
                model.to(device)
                #print("to acc")
                # Evaluate the model
                accuracy = evaluate_model(model, test_loader, device)
                #print(accuracy)
                # Store the result
                results[filename] = accuracy
                print(f"Model {filename} Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return results

def fix_dataset_shape(data):
    x = torch.stack([torch.from_numpy(data[i][0]) for i in range(len(data))])  # Inputs
    y = torch.tensor([data[i][1] for i in range(len(data))])  # Labels
    return x, y  # Returning both inputs and labels

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = "models"  # Path to the directory with saved models

X,Y = fix_dataset_shape(load_dataset("test_set.pkl","Datasets/"))
test_dataset = CustomDataset(X, Y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Load and evaluate all models in the directory

evaluate_all_models(models_dir, test_loader, device)