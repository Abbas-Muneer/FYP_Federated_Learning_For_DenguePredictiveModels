import pandas as pd
import os

# Define file paths
input_file_path = r'C:\Users\MSII\Desktop\FL_Draft1\Global.xlsx'
output_folder_path = r'C:\Users\MSII\Desktop\FL_Draft1'

# Read the Excel file
data = pd.read_excel(input_file_path)

# Get the number of rows and calculate the number of rows per client
num_rows = len(data)
rows_per_client = num_rows // 10

# Split the data into 10 parts and save each as a new Excel file
for i in range(10):
    start_idx = i * rows_per_client
    # Include all remaining rows in the last client
    end_idx = (i + 1) * rows_per_client if i < 9 else num_rows
    client_data = data.iloc[start_idx:end_idx]
    client_file_name = os.path.join(output_folder_path, f'client_{i + 1}.xlsx')
    client_data.to_excel(client_file_name, index=False)

print(f"Successfully split the data into 10 parts. Files saved in {output_folder_path}.")
