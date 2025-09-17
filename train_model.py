# train_model.py - replacement (copy & overwrite)
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILENAME = "flight_data.csv"  # produced by convert_excel_to_csv.py
MODEL_FILENAME = "model.joblib"

def load_data(path=DATA_FILENAME):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please run convert_excel_to_csv.py or ensure Data_Train.xlsx exists.")
    df = pd.read_csv(path)
    return df

def dur_to_minutes(s):
    # Accepts "2h 50m" or "50m" or "3h"
    try:
        s = str(s)
        hours = 0
        mins = 0
        if 'h' in s:
            parts = s.split('h')
            hours = int(parts[0].strip()) if parts[0].strip().isdigit() else 0
            if len(parts) > 1 and 'm' in parts[1]:
                mins = int(parts[1].replace('m','').strip()) if parts[1].replace('m','').strip().isdigit() else 0
        elif 'm' in s:
            mins = int(s.replace('m','').strip()) if s.replace('m','').strip().isdigit() else 0
        return hours*60 + mins
    except:
        return np.nan

def clean_and_engineer(df):
    df = df.copy()
    # Drop rows with no Price (target)
    if 'Price' in df.columns:
        df = df.dropna(subset=['Price'])
    # Date features
    if 'Date_of_Journey' in df.columns:
        df['Journey_Date'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce')
        df['Journey_Day'] = df['Journey_Date'].dt.day.fillna(0).astype(int)
        df['Journey_Month'] = df['Journey_Date'].dt.month.fillna(0).astype(int)
    # Duration to minutes
    if 'Duration' in df.columns:
        df['Duration_str'] = df['Duration'].astype(str).str.replace('.', '', regex=False)
        df['Duration_mins'] = df['Duration_str'].apply(dur_to_minutes)
    # Basic fill for categorical
    for c in ['Airline','Source','Destination','Total_Stops','Additional_Info']:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('Unknown')
    # Keep only columns we may want
    return df

def prepare_features(df):
    df = df.copy()
    feature_cols = []
    if 'Duration_mins' in df.columns:
        feature_cols.append('Duration_mins')
    for c in ['Journey_Day','Journey_Month']:
        if c in df.columns:
            feature_cols.append(c)
    # One-hot encode categorical columns present
    cat_cols = [c for c in ['Airline','Source','Destination','Total_Stops'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Build X and y
    X = df[[c for c in df.columns if c in feature_cols or c.startswith(tuple(cat_cols))]]
    # If no columns selected, fallback to numeric columns
    if X.shape[1] == 0:
        X = df.select_dtypes(include=[np.number])
    y = df['Price'] if 'Price' in df.columns else None
    return X, y

def train():
    print("Loading data...")
    df = load_data()
    print("Cleaning / engineering...")
    df = clean_and_engineer(df)
    X, y = prepare_features(df)
    if y is None:
        raise RuntimeError("No 'Price' column found in dataset.")
    print("Shape X:", X.shape)
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    # Save model + columns so app can re-create columns
    joblib.dump({'model': model, 'columns': X.columns.tolist()}, MODEL_FILENAME)
    print(f"Saved model to {MODEL_FILENAME}")

if __name__ == "__main__":
    train()
