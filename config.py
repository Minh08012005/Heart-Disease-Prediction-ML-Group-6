"""
Configuration file for Heart Disease Prediction project.

Contains all paths, hyperparameters, and settings.
"""

import os
from pathlib import Path

# ===== PROJECT PATHS =====
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_output"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_PROCESSED_DIR.mkdir(exist_ok=True)
DATA_RAW_DIR.mkdir(exist_ok=True)

# ===== DATA PATHS =====
RAW_DATA_PATH = DATA_DIR / "heart.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "heart_preprocessed.csv"

# ===== PREPROCESSING CONFIG =====
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "HeartDisease"

# Columns to scale (numeric)
NUMERIC_FEATURES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"
]

# Columns to encode (categorical)
CATEGORICAL_FEATURES = [
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"
]

# Invalid values handling
INVALID_VALUES_HANDLING = {
    "Cholesterol": {"invalid": 0, "strategy": "median_by_target"},
    "RestingBP": {"invalid": 0, "strategy": "median_global"}
}

# ===== MODEL HYPERPARAMETERS =====
MODELS_CONFIG = {
    "decision_tree": {
        "max_depth": 10,
        "min_samples_split": 20
    },
    "naive_bayes": {},
    "sklearn_decision_tree": {
        "max_depth": 10,
        "min_samples_split": 20,
        "random_state": RANDOM_STATE,
        "name": "Sklearn Decision Tree"
    },
    "sklearn_naive_bayes": {
        "name": "Sklearn Naive Bayes"
    },
    "svm": {
        "name": "SVM"
    },
    "knn": {
        "n_neighbors": 5,
        "name": "KNN"
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "name": "Random Forest"
    },
    "logistic_regression": {
        "random_state": RANDOM_STATE,
        "name": "Logistic Regression"
    }
}

# ===== EVALUATION CONFIG =====
METRICS = ["accuracy", "precision", "recall", "f1"]
K_FOLD_SPLITS = 5

# ===== LOGGING & OUTPUT =====
VERBOSE = True
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
MODEL_SAVE_FORMAT = "models_output/{model_name}_{timestamp}.pkl"

# ===== VISUALIZATION CONFIG =====
PLOT_DPI = 100
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Configuration loaded
