
import pandas as pd
import os

# This script looks for common Excel files from the original upload and converts the main training file to flight_data.csv
candidates = ['Data_Train.xlsx', 'Data_Train.csv', 'data_train.xlsx', 'train.xlsx']
found = None
for fname in candidates:
    if os.path.exists(fname):
        found = fname
        break

# also check in the project directory
project_root = os.path.dirname(__file__)
for fname in os.listdir(project_root):
    if fname.lower().endswith('.xlsx') and 'train' in fname.lower():
        found = fname
        break

if not found:
    print("No training Excel file found in project root. Please place your training file (e.g., Data_Train.xlsx) in the project root.")
else:
    df = pd.read_excel(os.path.join(project_root, found))
    out = os.path.join(project_root, 'flight_data.csv')
    df.to_csv(out, index=False)
    print(f"Converted {found} -> {out}")
