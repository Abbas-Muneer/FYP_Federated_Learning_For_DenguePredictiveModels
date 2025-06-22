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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os



class Net(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(Net, self).__init__()
        
        # 
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
    net.eval().to(device)
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features).squeeze()
            total_loss += criterion(outputs, labels).item()
            probs = torch.sigmoid(outputs)
            preds = torch.round(probs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Infected", "Infected"])
    disp.plot(cmap="Blues")
    os.makedirs("flower_graphs", exist_ok=True)
    plt.savefig("flower_graphs/confusion_matrix.png")
    plt.close()

    return total_loss, acc, prec, rec, f1