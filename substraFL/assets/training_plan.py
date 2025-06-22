
import torch
import torch.nn as nn
from model import Net 

class MyTrainingPlan:
    def __init__(self):
        self.model = Net()

    def training_step(self, dataloader, optimizer):
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        for features, labels in dataloader:
            optimizer.zero_grad()
            output = self.model(features)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()
        return loss.item()

    def validation_step(self, dataloader):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for features, labels in dataloader:
                outputs = self.model(features)
                predicted = torch.round(torch.sigmoid(outputs)).int()
                correct += (predicted == labels.int()).sum().item()
                total += labels.size(0)
        return correct / total

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
