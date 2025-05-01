import subprocess
from flask import Blueprint, request, jsonify, session
from flask_backend.models import Dataset, db, Client
import yaml
import datetime
import os
from flask import Blueprint, jsonify
from subprocess import Popen, PIPE

import datetime
import os
import pandas as pd
import torch
from flask import Blueprint, jsonify, send_file
from subprocess import Popen, PIPE
from torch.utils.data import DataLoader, TensorDataset, random_split
from flask_backend.models import Dataset, FLConfig, db
from model import Net, train, test 
from werkzeug.security import generate_password_hash, check_password_hash  

admin_blueprint = Blueprint("admin_routes", __name__)

#user login/signup stuffs
@admin_blueprint.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    is_admin = data.get("is_admin", False)  

    if not name or not email or not password:
        return jsonify({"error": "All fields are required."}), 400

    existing_user = Client.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "User already exists."}), 400

    new_user = Client(name=name, email=email, password=password)  
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully."}), 200

@admin_blueprint.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = Client.query.filter_by(email=email).first()
    if not user or user.password != password:  
        return jsonify({"error": "Invalid credentials."}), 401

    session["user_id"] = user.id
    session["user_name"] = user.name
    session["is_admin"] = user.is_admin  

    return jsonify({
        "message": "Login successful.",
        "user_id": user.id,
        "name": user.name,
        "is_admin": user.is_admin  
    }), 200

# Logout
@admin_blueprint.route('/logout', methods=['POST'])
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    return jsonify({"message": "Logged out successfully."}), 200

# Check Session
@admin_blueprint.route('/session', methods=['GET'])
def get_session():
    if "user_id" not in session:
        return jsonify({"user": None})

    user = Client.query.get(session["user_id"])
    if not user:
        return jsonify({"user": None})

    return jsonify({
        "user": {
            "id": user.id,
            "name": user.name,
            "is_admin": user.is_admin
        }
    })

@admin_blueprint.route('/admin/get-clients', methods=['GET'])
def get_clients():
    # Fetch all client details from the database
    datasets = Dataset.query.all()
    client_data = [
        {
            "id": dataset.id,
            "client_id": dataset.client_id,
            "dataset_name": dataset.dataset_name,
            "accuracy": dataset.accuracy,
            "loss": dataset.loss,
            "contribution_score": dataset.contribution_score
        }
        for dataset in datasets
    ]
    return jsonify({"clients": client_data}), 200

@admin_blueprint.route('/admin/update-config', methods=['POST'])
def update_config():
    data = request.get_json()
    config_file_path = r"C:\Users\MSII\Desktop\FL_Draft1\conf\base.yaml"  # Path to the configuration file

    try:
        # Load and update configuration
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)

        # Update the configuration
        config['config_fit']['local_epochs'] = int(data.get('epochs', config['config_fit']['local_epochs']))
        config['num_rounds'] = int(data.get('rounds', config['num_rounds']))
        config['config_fit']['lr'] = float(data.get('learning_rate', config['config_fit']['lr']))
        config['config_fit']['momentum'] = float(data.get('momentum', config['config_fit']['momentum']))

        # Save the updated configuration
        with open(config_file_path, "w") as file:
            yaml.dump(config, file)

        return jsonify({"message": "Configuration updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to update configuration: {str(e)}"}), 500
    

# Paths
FL_TRAINING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../main.py"))
outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs"))
client_datasets_dir = r"C:\Users\MSII\Desktop\FL_Draft1\client_datasets"  

@admin_blueprint.route('/admin/start-fl-training', methods=['POST'])
def start_fl_training():
    try:
        # 1️⃣ Load FL configuration
        config_file_path = r"C:\Users\MSII\Desktop\FL_Draft1\conf\base.yaml"
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)
        
        local_epochs = config['config_fit']['local_epochs']
        lr = config['config_fit']['lr']
        momentum = config['config_fit']['momentum']

        # Train each client locally using the same FL configurations
        datasets = Dataset.query.all()
        if not datasets:
            return jsonify({"error": "No client datasets found."}), 400

        client_results = {}

        for dataset in datasets:
            client_id = dataset.client_id
            dataset_filename = dataset.dataset_name
            dataset_path = os.path.join(client_datasets_dir, dataset_filename)  

            # Load dataset
            print("Loading dataset")
            df = pd.read_excel(dataset_path)
            features = df[["Fever", "Headache", "JointPain", "Bleeding"]].values
            target = df["Dengue"].values

            features_tensor = torch.tensor(features, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)

            # Create DataLoader
            dataset = TensorDataset(features_tensor, target_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

           

            # 
            class OldNet(torch.nn.Module):
                def __init__(self, num_classes: int = 1):
                    super(OldNet, self).__init__()
                    self.fc1 = torch.nn.Linear(4, 64)
                    self.dropout = torch.nn.Dropout(0.2)
                    self.fc2 = torch.nn.Linear(64, 32)
                    self.fc3 = torch.nn.Linear(32, num_classes)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            model = OldNet().to("cpu")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            criterion = torch.nn.BCEWithLogitsLoss()

            print("Training Client Model")
            model.train()
            for epoch in range(local_epochs):
                for features, labels in train_loader:
                    features, labels = features.to("cpu"), labels.to("cpu")
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs.squeeze(), labels)
                    loss.backward()
                    optimizer.step()

            print("Evaluating Client Model")
            model.eval()
            correct, loss = 0, 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to("cpu"), labels.to("cpu")
                    outputs = model(features)
                    loss += criterion(outputs.squeeze(), labels).item()
                    predicted = torch.round(torch.sigmoid(outputs)).int()
                    correct += (predicted == labels.int()).sum().item()
            
            client_accuracy = correct / len(val_loader.dataset)
            client_loss = loss / len(val_loader)

            print(f"🔹 {client_id} Accuracy: {client_accuracy}")
            print(f"🔹 {client_id} Loss: {client_loss}")


            #  Store in database
            print(f"🔹 Updating DB for {client_id} | Accuracy: {client_accuracy} | Loss: {client_loss}")
            client_results[client_id] = {"accuracy": client_accuracy, "loss": client_loss}

           #  Directly update the database using client_id
            db_dataset = Dataset.query.filter_by(client_id=client_id).first()

            if db_dataset:
                db_dataset.accuracy = client_accuracy
                db_dataset.loss = client_loss
                db.session.commit()
                print(f" Updated database: {client_id} -> Accuracy: {client_accuracy}, Loss: {client_loss}")
            else:
                print(f"⚠️ Warning: No dataset entry found for {client_id}")

        # 3️⃣ Start FL Training
        print("Starts FL training")
        process = Popen(["python", FL_TRAINING_PATH], cwd=os.path.dirname(FL_TRAINING_PATH), stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return jsonify({"error": f"FL training failed: {stderr.decode()}"}), 500

        # 4️⃣ Extract Accuracy & Loss from `main.py` output
        stdout_decoded = stdout.decode()
        print(stdout_decoded)  # Debugging: Print entire training output

        final_accuracy = None
        final_loss = None

        for line in stdout_decoded.split("\n"):
            if "FINAL_ACCURACY" in line:
                final_accuracy = float(line.split(":")[-1].strip())
            if "FINAL_LOSS" in line:
                final_loss = float(line.split(":")[-1].strip())

        if final_accuracy is None or final_loss is None:
            return jsonify({"error": "Failed to extract accuracy/loss from training output."}), 500
            print("error Failed to extract accuracy/loss from training output.")

        # 5️⃣ Compute Contribution Score
        print("Computing Contribution Score")
        for dataset in datasets:
            client_accuracy = client_results[dataset.client_id]["accuracy"]
            client_loss = client_results[dataset.client_id]["loss"]

            accuracy_contribution = client_accuracy / final_accuracy if final_accuracy > 0 else 0
            loss_contribution = final_loss / client_loss if client_loss > 0 else 0

            contribution_score = 0.5 * accuracy_contribution + 0.5 * loss_contribution
            print(contribution_score)

            # Store in DB
            print("Storing contribution score in DB")
            dataset.contribution_score = contribution_score
            db.session.commit()

        #  Save global model
        model_save_dir = os.path.join(outputs_dir, "Global_Models")
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, f"global_model.pth")


        # Ensure correct model saving
        global_model = Net(num_classes=1)  # Create a new instance of the model
        torch.save({"model": global_model.state_dict()}, model_path)  # Save model state_dict properly

        #  Update global accuracy & loss in DB
        #  Define timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("Storing Global Model Summary in DB")
        if not FLConfig.query.filter_by(key="global_model_accuracy").first():
            db.session.add(FLConfig(key="global_model_accuracy", value=str(final_accuracy)))

        if not FLConfig.query.filter_by(key="global_model_loss").first():
            db.session.add(FLConfig(key="global_model_loss", value=str(final_loss)))

        if not FLConfig.query.filter_by(key="global_model_last_trained").first():
            db.session.add(FLConfig(key="global_model_last_trained", value=timestamp))

        # If they already exist, update them
        FLConfig.query.filter_by(key="global_model_accuracy").update({"value": str(final_accuracy)})
        FLConfig.query.filter_by(key="global_model_loss").update({"value": str(final_loss)})
        FLConfig.query.filter_by(key="global_model_last_trained").update({"value": timestamp})
        db.session.commit()

        return jsonify({
            "message": "FL training completed.",
            "accuracy": final_accuracy,
            "loss": final_loss,
            "model_path": model_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 4. Fetch Summary
@admin_blueprint.route('/admin/summary', methods=['GET'])
def admin_summary():
    try:
        # Ensure global model metrics exist in the database
        default_values = {
            "global_model_accuracy": "N/A",
            "global_model_loss": "N/A",
            "global_model_last_trained": "N/A"
        }
        for key, value in default_values.items():
            entry = FLConfig.query.filter_by(key=key).first()
            if entry is None:
                db.session.add(FLConfig(key=key, value=value))
        db.session.commit()

        # Retrieve the latest training metrics
        global_accuracy = FLConfig.query.filter_by(key="global_model_accuracy").first().value
        global_loss = FLConfig.query.filter_by(key="global_model_loss").first().value
        last_trained = FLConfig.query.filter_by(key="global_model_last_trained").first().value

        return jsonify({
            "accuracy": global_accuracy,
            "loss": global_loss,
            "last_trained": last_trained
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

