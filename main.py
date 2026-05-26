"""
main.py - Main entry point for Heart Disease Prediction project.

This script runs the complete ML pipeline:
1. Load data
2. Preprocess & clean
3. Train models
4. Evaluate on test set
5. Generate comparison report
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Import configuration
import config
from src.data import load_train_test
from src.utils import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.models.decision_tree import DecisionTree
from src.models.naive_bayes import NaiveBayes

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️  Sklearn not available - will use custom models only")


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return metrics


def print_metrics_table(all_metrics):
    """Print metrics table."""
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 78)
    
    for metrics in all_metrics:
        name = metrics['name'][:28]
        print(f"{name:<30} {metrics['accuracy']:.4f}     "
              f"{metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}")


def main():
    """
    Main pipeline execution.
    """
    print("\n")
    print("=" * 70)
    print("  HEART DISEASE PREDICTION PROJECT")
    print("  Complete ML Pipeline Execution")
    print("=" * 70)
    
    # ===== STEP 1: LOAD DATA =====
    print_header("STEP 1: Load Data")
    
    logger.info(f"[INFO] Loading preprocessed data from: {config.PREPROCESSED_DATA_PATH}")
    X_train, X_test, y_train, y_test = load_train_test(
        config.PREPROCESSED_DATA_PATH,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    logger.info(f"[OK] Data loaded successfully!")
    logger.info(f"   - Train set: {len(X_train)} samples")
    logger.info(f"   - Test set: {len(X_test)} samples")
    logger.info(f"   - Features: {X_train.shape[1]}")
    
    # ===== STEP 2: TRAIN MODELS =====
    print_header("STEP 2: Train Models")
    
    models_to_train = {}
    results = []
    
    # Custom Decision Tree
    logger.info("[TRAIN] Training Custom Decision Tree...")
    dt_custom = DecisionTree(**config.MODELS_CONFIG['decision_tree'])
    dt_custom.fit(X_train, y_train)
    models_to_train['custom_dt'] = dt_custom
    metrics = evaluate_model(dt_custom, X_test, y_test, 
                           "Custom Decision Tree")
    results.append(metrics)
    logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    # Custom Naive Bayes
    logger.info("[TRAIN] Training Custom Naive Bayes...")
    nb_custom = NaiveBayes(**config.MODELS_CONFIG['naive_bayes'])
    nb_custom.fit(X_train, y_train)
    models_to_train['custom_nb'] = nb_custom
    metrics = evaluate_model(nb_custom, X_test, y_test,
                           "Custom Naive Bayes")
    results.append(metrics)
    logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
    
    # Sklearn models (if available)
    if SKLEARN_AVAILABLE:
        logger.info("[TRAIN] Training Sklearn models...")
        
        sklearn_models = [
            (DecisionTreeClassifier(**{k: v for k, v in config.MODELS_CONFIG['sklearn_decision_tree'].items() if k != 'name'}),
             config.MODELS_CONFIG['sklearn_decision_tree']['name']),
            (GaussianNB(), config.MODELS_CONFIG['sklearn_naive_bayes']['name']),
            (SVC(), config.MODELS_CONFIG['svm']['name']),
            (KNeighborsClassifier(n_neighbors=config.MODELS_CONFIG['knn']['n_neighbors']),
             config.MODELS_CONFIG['knn']['name']),
            (RandomForestClassifier(**{k: v for k, v in config.MODELS_CONFIG['random_forest'].items() if k != 'name'}),
             config.MODELS_CONFIG['random_forest']['name']),
            (LogisticRegression(**{k: v for k, v in config.MODELS_CONFIG['logistic_regression'].items() if k != 'name'}),
             config.MODELS_CONFIG['logistic_regression']['name']),
        ]
        
        for model, name in sklearn_models:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, name)
            results.append(metrics)
            logger.info(f"   → {name}: {metrics['accuracy']:.4f}")
    
    # ===== STEP 3: COMPARE RESULTS =====
    print_header("STEP 3: Model Comparison")
    
    print_metrics_table(results)
    
    # Find best model
    best_model = max(results, key=lambda x: x['f1'])
    logger.info(f"\n[BEST] Best model: {best_model['name']}")
    logger.info(f"   - Accuracy:  {best_model['accuracy']:.4f}")
    logger.info(f"   - Precision: {best_model['precision']:.4f}")
    logger.info(f"   - Recall:    {best_model['recall']:.4f}")
    logger.info(f"   - F1-Score:  {best_model['f1']:.4f}")
    
    # ===== STEP 4: CONFUSION MATRIX (Custom models) =====
    print_header("STEP 4: Confusion Matrix Analysis")
    
    for name, model in [('Custom Decision Tree', dt_custom), ('Custom Naive Bayes', nb_custom)]:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n{name}:")
        logger.info(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        logger.info(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # ===== STEP 5: SUMMARY =====
    print_header("STEP 5: Summary")
    
    logger.info(f"\n[SUMMARY] Pipeline Execution Summary:")
    logger.info(f"   [OK] Data loaded: {len(X_train) + len(X_test)} samples")
    logger.info(f"   [OK] Models trained: {len(results)} models")
    logger.info(f"   [OK] Best model: {best_model['name']} (F1: {best_model['f1']:.4f})")
    logger.info(f"   [OK] Evaluation complete!")
    
    print("\n" + "=" * 70)
    print("  [OK] All steps completed successfully!")
    print("=" * 70 + "\n")
    
    return results, models_to_train


if __name__ == "__main__":
    try:
        results, models = main()
    except Exception as e:
        logger.error(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
