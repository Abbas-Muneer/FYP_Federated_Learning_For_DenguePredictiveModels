# import pandas as pd
# import os

# # Define file paths
# input_file_path = r'C:\Users\MSII\Desktop\FL_Draft1\Global.xlsx'
# output_folder_path = r'C:\Users\MSII\Desktop\FL_Draft1'

# # Read the Excel file
# data = pd.read_excel(input_file_path)

# # Get the number of rows and calculate the number of rows per client
# num_rows = len(data)
# rows_per_client = num_rows // 10

# # Split the data into 10 parts and save each as a new Excel file
# for i in range(10):
#     start_idx = i * rows_per_client
#     # Include all remaining rows in the last client
#     end_idx = (i + 1) * rows_per_client if i < 9 else num_rows
#     client_data = data.iloc[start_idx:end_idx]
#     client_file_name = os.path.join(output_folder_path, f'client_{i + 1}.xlsx')
#     client_data.to_excel(client_file_name, index=False)

# print(f"Successfully split the data into 10 parts. Files saved in {output_folder_path}.")


# import pandas as pd

# # Load the Excel file
# file_path = r'C:\Users\MSII\Desktop\FL_Draft1\dengue_dataset.xlsx'
# df = pd.read_excel(file_path)

# # Print number of columns and rows
# num_rows, num_columns = df.shape
# print(f"Number of rows: {num_rows}")
# print(f"Number of columns: {num_columns}")

# # Count True and False in "Infected" column
# infected_counts = df['Infected'].value_counts()

# num_true = infected_counts.get(True, 0)
# num_false = infected_counts.get(False, 0)

# print(f"Number of True in 'Infected': {num_true}")
# print(f"Number of False in 'Infected': {num_false}")


# # Separate True and False
# df_true = df[df['Infected'] == True]
# df_false = df[df['Infected'] == False]

# # Undersample the majority class (False) to match minority class size
# df_false_sampled = df_false.sample(n=len(df_true), random_state=42)

# # Combine to make balanced DataFrame
# df_balanced = pd.concat([df_true, df_false_sampled])

# # Shuffle the rows
# df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# # Print new counts
# balanced_counts = df_balanced['Infected'].value_counts()
# print(f"Number of True in 'Infected': {balanced_counts.get(True, 0)}")
# print(f"Number of False in 'Infected': {balanced_counts.get(False, 0)}")

# # Optionally, save the balanced data to a new file
# df_balanced.to_excel(r'C:\Users\MSII\Desktop\FL_Draft1\dengue_dataset_balanced.xlsx', index=False)


# import pandas as pd

# # Load data
# file_path = r'C:\Users\MSII\Desktop\FL_Draft1\dengue_dataset_balanced.xlsx'
# df = pd.read_excel(file_path)

# # Duplicate rows
# df_doubled = pd.concat([df, df]).reset_index(drop=True)

# # Print new shape
# num_rows, num_columns = df_doubled.shape
# print(f"New number of rows: {num_rows}")
# print(f"Number of columns: {num_columns}")

# # Overwrite the original file
# df_doubled.to_excel(file_path, index=False)

# print("Dataset duplicated and overwritten successfully.")

import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import os

# ⚙️ Settings
NUM_CLIENTS = 5
DATA_PATH = r"C:\Users\MSII\Desktop\FL_Draft1\dengue_dataset_balanced.xlsx"
OUTPUT_DIR = "substraFL/scripts/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🧪 Load and preprocess
df = pd.read_excel(DATA_PATH)
features = df[['Temperature', 'Platelet_Count', 'White_Blood_Cell_Count', 'Body_Pain', 'Rash']]
labels = df['Infected'].astype(int)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = torch.tensor(features_scaled, dtype=torch.float32)
y = torch.tensor(labels.values, dtype=torch.float32)

# 🧠 Stratified split using sklearn
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.0)  # placeholder
indices = list(range(len(y)))
labels_np = labels.values

# Generate balanced client splits
client_data = [[] for _ in range(NUM_CLIENTS)]
client_labels = [[] for _ in range(NUM_CLIENTS)]

for label in [0, 1]:  # loop over each class
    class_indices = [i for i, v in enumerate(labels_np) if v == label]
    split = StratifiedShuffleSplit(n_splits=1, test_size=0, train_size=1.0)
    class_chunks = torch.chunk(torch.tensor(class_indices), NUM_CLIENTS)
    for i in range(NUM_CLIENTS):
        for idx in class_chunks[i]:
            client_data[i].append(X[idx])
            client_labels[i].append(y[idx])

# 💾 Save partitioned data
for i in range(NUM_CLIENTS):
    x_tensor = torch.stack(client_data[i])
    y_tensor = torch.stack(client_labels[i])
    torch.save({"x": x_tensor, "y": y_tensor}, os.path.join(OUTPUT_DIR, f"client_{i}.pt"))
    print(f"✅ Saved client_{i}.pt with {len(x_tensor)} samples.")

