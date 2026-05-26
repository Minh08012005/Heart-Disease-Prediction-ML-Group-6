"""
Data preprocessing module.

Functions for data cleaning, encoding, scaling, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(df, numeric_features, categorical_features, target_col="HeartDisease"):
    """
    Preprocess data using StandardScaler and OneHotEncoder.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        target_col: Target column name
        
    Returns:
        tuple: (X_processed, y, preprocessor) where preprocessor is the fitted transformer
    """
    
    # Separate target
    y = df[target_col].values
    X = df.drop(target_col, axis=1)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_features),
            ('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    print(f"✅ Preprocessing completed")
    print(f"   Original features: {X.shape[1]}")
    print(f"   Processed features: {X_processed.shape[1]}")
    
    return X_processed, y, preprocessor


def handle_invalid_values(df, config):
    """
    Handle invalid values (e.g., 0 values in Cholesterol and RestingBP).
    
    Args:
        df: Input DataFrame
        config: Dictionary with invalid value handling configuration
        
    Returns:
        pd.DataFrame: DataFrame with cleaned values
    """
    df_clean = df.copy()
    
    for col, settings in config.items():
        if col not in df_clean.columns:
            continue
            
        invalid_val = settings.get("invalid", 0)
        strategy = settings.get("strategy", "median_global")
        
        # Find invalid values
        invalid_mask = df_clean[col] == invalid_val
        n_invalid = invalid_mask.sum()
        
        if n_invalid == 0:
            continue
        
        print(f"⚠️  Column '{col}': Found {n_invalid} invalid values (={invalid_val})")
        
        if strategy == "median_global":
            replacement = df_clean[col][df_clean[col] != invalid_val].median()
            df_clean.loc[invalid_mask, col] = replacement
            print(f"   → Replaced with global median: {replacement:.2f}")
            
        elif strategy == "median_by_target":
            # This requires target column to be available
            if "HeartDisease" in df_clean.columns:
                for target_val in [0, 1]:
                    target_mask = (df_clean["HeartDisease"] == target_val) & invalid_mask
                    if target_mask.any():
                        replacement = df_clean[col][
                            (df_clean["HeartDisease"] == target_val) & (df_clean[col] != invalid_val)
                        ].median()
                        df_clean.loc[target_mask, col] = replacement
                        print(f"   → Replaced (target={target_val}) with median: {replacement:.2f}")
    
    return df_clean
