import torchvision
from torchvision import transforms
import torch
import os
import pickle

NUM_CLIENTS = 2
SAVE_PATH = "data"

os.makedirs(SAVE_PATH, exist_ok=True)

# Downloaddataset
train_set = torchvision.datasets.MNIST(root=SAVE_PATH, train=True, download=True, transform=transforms.ToTensor())

# Split dataset
data_per_client = len(train_set) // NUM_CLIENTS
splits = torch.utils.data.random_split(train_set, [data_per_client] * NUM_CLIENTS)

# Save each client's data separately
for i, subset in enumerate(splits):
    images, labels = zip(*[subset[j] for j in range(len(subset))])
    torch.save({"x": torch.stack(images), "y": torch.tensor(labels)}, os.path.join(SAVE_PATH, f"client_{i}.pt"))

print(f"Data prepared for {NUM_CLIENTS} clients in '{SAVE_PATH}/'")
