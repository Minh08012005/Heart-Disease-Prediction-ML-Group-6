"""
scripts/evaluate.py - Model evaluation script.

Loads a trained model and evaluates it on test data.
Usage: python scripts/evaluate.py --model models_output/decision_tree_*.pkl
"""

import argparse
import pickle
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.data import load_train_test
from src.utils import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model_path):
    """
    Load model and evaluate on test data.
    
    Args:
        model_path: Path to trained model
    """
    # Load model
    logger.info(f"📂 Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded")
    
    # Load data
    logger.info("📊 Loading test data...")
    X_train, X_test, y_train, y_test = load_train_test(
        config.PREPROCESSED_DATA_PATH,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # Predict
    logger.info("🔮 Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n Metrics:")
    logger.info(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"   Precision: {prec:.4f}")
    logger.info(f"   Recall:    {rec:.4f}")
    logger.info(f"   F1-Score:  {f1:.4f}")
    
    logger.info(f"\n Confusion Matrix:")
    logger.info(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
    logger.info(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    logger.info(f"\n Interpretation:")
    if cm[1,0] > 0:
        fnr = cm[1,0] / (cm[1,0] + cm[1,1])
        logger.info(f"   False Negative Rate: {fnr:.2%} (missed {cm[1,0]} diseased)")
    if cm[0,1] > 0:
        fpr = cm[0,1] / (cm[0,0] + cm[0,1])
        logger.info(f"   False Positive Rate: {fpr:.2%} (false alarms: {cm[0,1]})")
    
    logger.info("\n" + "=" * 60)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained heart disease prediction model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pkl file)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        logger.error(f"❌ Model file not found: {args.model}")
        return
    
    logger.info("=" * 60)
    logger.info("🫀 Heart Disease Prediction - Evaluation Script")
    logger.info("=" * 60)
    
    metrics = evaluate_model(args.model)


if __name__ == "__main__":
    main()
