import torch
import os
from model import Net, train, test
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Hyperparameters
NUM_CLIENTS = 10
LOCAL_EPOCHS = 30
ROUNDS = 10
BATCH_SIZE = 32
device = "cpu"
LR = 0.01
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

def fed_avg(state_dicts):
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
    return avg_state


def load_client_dataset(client_id):
    path = f"scripts/scripts/data/client_{client_id}.pt"
    data = torch.load(path)
    dataset = TensorDataset(data["x"], data["y"])

    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


client_loaders = []
test_loaders = []
for i in range(NUM_CLIENTS):
    train_loader, test_loader = load_client_dataset(i)
    client_loaders.append(train_loader)
    test_loaders.append(test_loader)

global_model = Net().to(device)

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for round_num in range(1, ROUNDS + 1):
    print(f"\nRound {round_num}")
    client_states = []
    round_train_loss = 0

    for i, loader in enumerate(client_loaders):
        print(f"Training client {i}")
        model = Net()
        model.load_state_dict(global_model.state_dict())
        #optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        train_loss = train(model, loader, optimizer, LOCAL_EPOCHS, device)
        round_train_loss += train_loss
        client_states.append(model.state_dict())

    avg_train_loss = round_train_loss / NUM_CLIENTS
    train_losses.append(avg_train_loss)

    print("Aggregating models via FedAvg")
    global_state = fed_avg(client_states)
    global_model.load_state_dict(global_state)

    # Train Accuracy
    with torch.no_grad():
        global_model.eval()
        preds, labels = [], []
        for x_batch, y_batch in client_loaders[0]:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = global_model(x_batch).squeeze()
            pred = torch.round(torch.sigmoid(output))
            preds.extend(pred.cpu().numpy())
            labels.extend(y_batch.cpu().numpy())
        train_acc = accuracy_score(labels, preds)
        train_accuracies.append(train_acc)

    # Test Metrics on all clients
    test_loss, test_acc, prec, rec, f1 = test(global_model, test_loaders, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"→ Evaluation after Round {round_num}")
    print(f"   Train Loss : {avg_train_loss:.4f}")
    print(f"   Train Acc  : {train_acc:.4f}")
    print(f"   Test Loss  : {test_loss:.4f}")
    print(f"   Accuracy   : {test_acc:.4f}")
    print(f"   Precision  : {prec:.4f}")
    print(f"   Recall     : {rec:.4f}")
    print(f"   F1-Score   : {f1:.4f}")

print("\nFederated training complete.")

# Plotting
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

plt.figure()
plt.plot(range(1, ROUNDS + 1), test_accuracies, label="Test Accuracy", color="blue")
plt.plot(range(1, ROUNDS + 1), train_accuracies, label="Train Accuracy", color="orange")
plt.title("Accuracy over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png")
plt.close()
