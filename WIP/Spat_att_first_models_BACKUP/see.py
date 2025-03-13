import torch
from torchinfo import summary
import os

# Load model
MODEL_PATH = "spatial_attention_first_53_6845222.pth"  # Change this to your model path

if __name__ == "__main__":
    model = torch.load(MODEL_PATH,weights_only = False,map_location=torch.device('cpu'))
    #print(model)
    summary(model, input_size=(1, 22, 1125))  # Change input size as needed