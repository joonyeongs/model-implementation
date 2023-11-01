# Description: Neural Network Language Model in PyTorch
# conda install -c anaconda chardet
# conda install -c conda-forge portalocker
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
if __name__ == "__main__":
    #model = NeuralNetwork().to(device)
    #print(model)
    train_set, valid_set, test_set = WikiText2()

    for i, text in enumerate(train_set):
        if i == 5: break
        print(f'[{i}]: {text}')
