import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Net, train, test
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="base", version_base=None)
def client_contribution(cfg: DictConfig):
    #  Load the client dataset
    client_data_path = r"C:\Users\MSII\Desktop\FL_Draft1\client_2.xlsx"  # Replace with the actual path
    df = pd.read_excel(client_data_path)

    # Prepare the dataset (features and target)
    features = df[["Fever", "Headache", "JointPain", "Bleeding"]].values
    target = df["Dengue"].values

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32) 

    #  Create dataset and dataloaders
    dataset = TensorDataset(features_tensor, target_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    #  Initialize the model
    num_classes = cfg.num_classes
    device = torch.device("cpu")
    model = Net(num_classes=num_classes).to(device)

    
    learning_rate = cfg.config_fit.lr
    momentum = cfg.config_fit.momentum
    epochs = cfg.config_fit.local_epochs * cfg.num_rounds
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    #
    train(model, train_loader, optimizer, epochs, device)

    # Step 7: Evaluate the model
    loss, accuracy = test(model, val_loader, device)

    # Step 8: Output results
    print(f"Client 2 Contribution: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")



    # Weights for accuracy and loss contributions
    w1, w2 = 0.5, 0.5  # Equal weights for now, can be adjusted



if __name__ == "__main__":
    client_contribution()
