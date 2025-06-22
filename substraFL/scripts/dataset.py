import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.calibration import LabelEncoder

def get_csv_data(data_path: str = r"C:\Users\MSII\Desktop\FL_Draft1\dengue_dataset_balanced.xlsx"):
    # Load the dataset
    data = pd.read_excel(data_path)

    # Convert categorical columns to numerical
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])  
    
    # 
    X = data[['Temperature', 'Platelet_Count', 'White_Blood_Cell_Count', 'Body_Pain', 'Rash']]
    y = data['Infected'].astype(int)  # Binary target

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    return X_tensor, y_tensor

def save_partitioned_clients(num_clients=2, output_dir="scripts/data"):
    os.makedirs(output_dir, exist_ok=True)
    X, y = get_csv_data()

    sss = StratifiedShuffleSplit(n_splits=num_clients, test_size=1/num_clients, random_state=42)
    splits = list(sss.split(X, y))

    for i, (train_idx, _) in enumerate(splits):
        x_data = X[train_idx]
        y_data = y[train_idx]
        torch.save({"x": x_data, "y": y_data}, os.path.join(output_dir, f"client_{i}.pt"))
        print(f"\ Saved client_{i}.pt with {len(x_data)} samples.")

if __name__ == "__main__":
    save_partitioned_clients(num_clients=10)
