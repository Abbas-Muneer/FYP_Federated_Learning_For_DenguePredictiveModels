import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import flwr as fl
from typing import Dict, Union
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from typing import Dict, Union



class Net(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(Net, self).__init__()
        
        # Fully connected layers for the new dataset
        self.fc1 = nn.Linear(6, 64)  # 6 input features
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)  # Binary classification output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str):
    criterion = torch.nn.BCEWithLogitsLoss()
    net.train()
    net.to(device)
    for epoch in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            features, labels = data[0].to(device), data[1].to(device)
            outputs = net(features)
            loss += criterion(outputs.squeeze(), labels).item()
            predicted = torch.round(torch.sigmoid(outputs)).int()
            correct += (predicted == labels.int()).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy