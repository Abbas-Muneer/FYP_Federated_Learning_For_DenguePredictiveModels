import os
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

def create_client_folder(base_path="."):
    """Create a directory for storing client datasets."""
    client_folder = Path(base_path) / "client_datasets"
    client_folder.mkdir(exist_ok=True)
    return client_folder

def analyze_dataset(file_path):
    """Analyze the dataset to ensure it has the required columns."""
    required_columns = {"Name", "Fever", "Headache", "JointPain", "Bleeding", "Dengue"}
    
    try:
        df = pd.read_excel(file_path)
        if set(df.columns) >= required_columns:
            print("Dataset analysis complete: Valid dataset.")
            return df
        else:
            print("Dataset analysis failed: Invalid columns.")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def store_dataset(dataset, client_id, output_folder):
    """Store the dataset for a specific client."""
    client_file_path = output_folder / f"client_{client_id}.xlsx"
    dataset.to_csv(client_file_path, index=False)
    print(f"Dataset for client {client_id} stored at {client_file_path}.")

def combine_datasets(output_folder, combined_file_name="Updated_Dataset.xlsx"):
    """Combine all client datasets into one and save as a single file."""
    combined_dataset = pd.DataFrame()
    for file in output_folder.glob("client_*.csv"):
        df = pd.read_csv(file)
        combined_dataset = pd.concat([combined_dataset, df], ignore_index=True)

    combined_file_path = output_folder / combined_file_name
    combined_dataset.to_excel(combined_file_path, index=False)
    print(f"Combined dataset saved at {combined_file_path}.")

@hydra.main(config_path="conf", config_name="base", version_base=None)
def process_datasets(cfg: DictConfig):
    """Main function to process datasets for all clients."""
    num_clients = cfg.num_clients

    # Create a folder for client datasets
    client_dataset_folder = create_client_folder()

    for client_id in range(1, num_clients + 1):
        print(f"\nProcessing dataset for client {client_id}...")
        file_path = input(f"Enter the file path for client {client_id}: ").strip()

        # Analyze the dataset
        dataset = analyze_dataset(file_path)
        if dataset is not None:
            store_dataset(dataset, client_id, client_dataset_folder)
        else:
            print(f"Invalid dataset for client {client_id}. Please try again.")
            return

    # Combine all client datasets into one
    combine_datasets(client_dataset_folder)

if __name__ == "__main__":
    process_datasets()
