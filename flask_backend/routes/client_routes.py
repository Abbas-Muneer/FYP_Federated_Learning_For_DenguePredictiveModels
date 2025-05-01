import os
import tempfile
import pandas as pd
from flask import Blueprint, request, jsonify, send_file
from flask_backend.models import Dataset, db

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Net, train, test

client_blueprint = Blueprint("client_routes", __name__)

def analyze_dataset(file_path):
    """Analyze the dataset to ensure it has the required columns and compute dummy metrics."""
    required_columns = {"Name", "Fever", "Headache", "JointPain", "Bleeding", "Dengue"}

    try:
        # Load the dataset
        df = pd.read_excel(file_path)

        # Ensure all required columns are present
        if not required_columns.issubset(set(df.columns)):
            print("Dataset analysis failed: Missing required columns.")
            return False, None, None

        
        accuracy = 0  
        loss = 0           

        print(f"Dataset analysis complete: Valid dataset. Accuracy = {accuracy}, Loss = {loss}")
        return True, accuracy, loss

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False, None, None

@client_blueprint.route('/client/add-dataset', methods=['POST'])
def add_dataset():
    client_id = request.form.get("client_id", "Unknown_Client")
    dataset_file = request.files.get('file')

    # Save the file temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, dataset_file.filename)
    dataset_file.save(temp_path)

    # Analyze dataset
    valid, accuracy, loss = analyze_dataset(temp_path)
    if not valid:
        return jsonify({"error": "Invalid dataset. Please upload a valid dataset."}), 400

    # Store in database
    new_dataset = Dataset(client_id=client_id, dataset_name=dataset_file.filename, accuracy=accuracy, loss=loss)
    db.session.add(new_dataset)
    db.session.commit()

    return jsonify({"data": "Dataset added successfully.", "accuracy": accuracy, "loss": loss}), 200


# 3. Download Global Model
@client_blueprint.route('/client/download-model', methods=['GET'])
def download_model():
    try:
        outputs_dir = "outputs"
        
        # ✅ Find latest date directory
        date_dirs = sorted(os.listdir(outputs_dir), reverse=True)
        latest_date_dir = os.path.join(outputs_dir, date_dirs[0])

        # ✅ Find latest time directory
        time_dirs = sorted(os.listdir(latest_date_dir), reverse=True)
        latest_time_dir = os.path.join(latest_date_dir, time_dirs[0])

        model_path = r"C:\Users\MSII\Desktop\FL_Draft1\outputs\Global_Models\global_model.pth"

        if not os.path.exists(model_path):
            return jsonify({"error": "Global model not found."}), 404

        return send_file(model_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 4. Fetch Client Summary
@client_blueprint.route('/client/summary', methods=['GET'])
def client_summary():
    client_id = "Client_6"  
    dataset = Dataset.query.filter_by(client_id=client_id).first()
    if dataset:
        return jsonify({
            "client_id": client_id,
            "accuracy": dataset.accuracy,
            "loss": dataset.loss,
            "contribution_score": dataset.contribution_score
        }), 200
    return jsonify({"error": "Client data not found."}), 404