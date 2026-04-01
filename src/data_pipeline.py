# src/data_pipeline.py
import sys
import os

# 1. FORCE PYTHON TO FIND THE 'src' FOLDER
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 2. NOW WE CAN SAFELY IMPORT
import pandas as pd
import numpy as np
from src.config import FILE_NQ, FILE_BTC

def load_csv(path, suffix):
    """
    Loads a CSV, standardizes column names, and aligns the datetime 
    index to Warsaw time (CET/CEST) regardless of the data source.
    """
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [c.strip().lower() for c in df.columns]

    if 'open time' in df.columns:
        df = df.rename(columns={'open time': 'time'})
        tz_source = 'UTC'  
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'time'})
        tz_source = 'America/Chicago' 
    else:
        raise ValueError(f"No valid time column found for {suffix}. Check CSV headers.")

    if 'close time' in df.columns:
        df = df.drop(columns=['close time'])

    rename_dict = {c: f'{c}_{suffix}' for c in ['open', 'high', 'low', 'close'] if c in df.columns}
    df = df.rename(columns=rename_dict)

    df['datetime'] = pd.to_datetime(df['time'])
    df['datetime'] = df['datetime'].dt.tz_localize(tz_source, ambiguous='NaT', nonexistent='NaT')
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/Warsaw')
    df['datetime'] = df['datetime'].dt.tz_localize(None)

    keep = ['datetime', f'open_{suffix}', f'high_{suffix}', f'low_{suffix}', f'close_{suffix}']
    df = df.dropna(subset=['datetime']) 
    df = df[keep].set_index('datetime').sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df

def get_aligned_raw_data(how='inner'):
    """Loads both datasets and joins them."""
    print("Loading NQ data...")
    df_nq  = load_csv(FILE_NQ,  'nq')
    print("Loading BTC data...")
    df_btc = load_csv(FILE_BTC, 'btc')

    print(f"Merging data with '{how}' join...")
    df = df_btc.join(df_nq, how=how)
    
    df['ret_btc']   = df['close_btc'].pct_change()
    df['ret_nq']    = df['close_nq'].pct_change()
    df['hour']      = df.index.hour
    df['minute']    = df.index.minute
    df['date']      = df.index.date
    df['dayofweek'] = df.index.dayofweek
    
    return df

def apply_strategy_features(df):
    """Prepares raw, numeric features for the simulation engine."""
    df = df.copy()
    
    df = df[df['dayofweek'] < 5]          
    df.dropna(subset=['close_nq', 'close_btc'], inplace=True)

    prev_close_nq  = df['close_nq'].shift(1)
    df['gap_pct']  = (df['open_nq'] - prev_close_nq).abs() / prev_close_nq

    time_gap_min = df.index.to_series().diff().dt.total_seconds().div(60).fillna(0).values
    df['is_session_open'] = time_gap_min > 60

    df.dropna(inplace=True)
    return df

def get_train_test_split(df, split_ratio=0.70):
    """Splits the data chronologically."""
    split_i   = int(len(df) * split_ratio)
    train_df  = df.iloc[:split_i].copy()
    test_df   = df.iloc[split_i:].copy()
    
    return train_df, test_df

# ==========================================
# THIS ALLOWS YOU TO RUN THE FILE DIRECTLY
# ==========================================
if __name__ == "__main__":
    print("--- Testing Data Pipeline ---")
    raw_df = get_aligned_raw_data(how='inner')
    final_df = apply_strategy_features(raw_df)
    
    print("\nPipeline execution successful!")
    print(f"Total rows prepared: {len(final_df)}")
    print("\nFirst 5 rows:")
    print(final_df.head())