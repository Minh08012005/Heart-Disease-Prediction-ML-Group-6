"""
Data loading module.

Functions to load raw and preprocessed data from various sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath):
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[OK] Loaded {len(df)} samples from {filepath}")
    return df


def load_preprocessed_data(filepath):
    """
    Load preprocessed data.
    
    Args:
        filepath: Path to preprocessed CSV file
        
    Returns:
        tuple: (X, y, df) where X is features, y is target, df is full dataframe
    """
    df = load_data(filepath)
    
    # Assume last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"   Target distribution: {np.bincount(y.astype(int))}")
    
    return X, y, df


def load_train_test(filepath, target_col="HeartDisease", test_size=0.2, random_state=42):
    """
    Load data and split into train/test sets.
    
    Args:
        filepath: Path to data file
        target_col: Target column name
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from src.utils import train_test_split
    
    df = load_data(filepath)
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"[OK] Train/Test split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test
