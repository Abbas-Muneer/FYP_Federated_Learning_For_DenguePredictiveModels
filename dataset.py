import numpy as np
from sklearn.calibration import LabelEncoder
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split


def get_csv_data(data_path: str = r'C:\Users\MSII\Desktop\FL_Draft1\dengue_dataset_balanced.xlsx'):
    # Load the dataset
    data = pd.read_excel(data_path)
    
    # Convert categorical columns to numerical
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])  
    
    # Separate features and target
    X = data[['Temperature', 'Platelet_Count', 'White_Blood_Cell_Count', 'Body_Pain', 'Rash', 'Gender']]
    y = data['Infected'].astype(int)  
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    
    return X_tensor, y_tensor


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    X_tensor, y_tensor = get_csv_data()

    # Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )

    # Partition train into clients
    full_train_ds = TensorDataset(X_train, y_train)
    partition_len = [len(full_train_ds) // num_partitions] * num_partitions
    partition_len[-1] += len(full_train_ds) - sum(partition_len)
    trainsets = random_split(full_train_ds, partition_len, generator=torch.Generator().manual_seed(2023))

    trainloaders = []
    valloaders = []

    for trainset in trainsets:
        num_total = len(trainset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    # 
    test_ds = TensorDataset(X_test, y_test)
    testloader = DataLoader(test_ds, batch_size=128, shuffle=False)

    return trainloaders, valloaders, testloader