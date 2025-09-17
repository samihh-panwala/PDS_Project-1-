
import pandas as pd
import numpy as np

DATA_FILENAME = "flight_data.csv"  # update if your dataset has another name

def load_data(path=DATA_FILENAME):
    df = pd.read_csv(path)
    return df

def clean_and_engineer(df):
    # Basic cleaning: drop na target rows, parse dates, extract features
    df = df.copy()
    # common column names: 'Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops', 'Airline', 'Source', 'Destination', 'Price'
    # Make code robust by checking existence
    if 'Price' in df.columns:
        df = df.dropna(subset=['Price'])
    # Parse journey date if present
    if 'Date_of_Journey' in df.columns:
        df['Journey_Date'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce')
        df['Journey_Day'] = df['Journey_Date'].dt.day
        df['Journey_Month'] = df['Journey_Date'].dt.month
    # Duration: convert to minutes
    if 'Duration' in df.columns:
        def dur_to_min(x):
            try:
                parts = x.split()
                total = 0
                for p in parts:
                    if 'h' in p:
                        total += int(p.replace('h',''))*60
                    elif 'm' in p:
                        total += int(p.replace('m',''))
                return total
            except:
                return np.nan
        df['Duration_mins'] = df['Duration'].astype(str).str.replace(' ','').apply(lambda x: 
                        int(x.split('h')[0])*60 + int(x.split('h')[1].replace('m','')) if 'h' in x and 'm' in x else
                        (int(x.replace('m','')) if 'm' in x else (int(x.replace('h',''))*60 if 'h' in x else np.nan))
                        )
    # Simple encoding for categorical columns
    cat_cols = ['Airline','Source','Destination','Total_Stops']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('Unknown')
    # Drop rows with essential NaNs
    df = df.dropna(subset=[c for c in ['Duration_mins'] if c in df.columns])
    return df

if __name__ == '__main__':
    print("This module provides functions load_data() and clean_and_engineer()")
