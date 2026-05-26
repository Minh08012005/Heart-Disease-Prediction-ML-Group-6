"""
scripts/train.py - Model training script.

Trains a specific model and saves it for later use.
Usage: python scripts/train.py --model decision_tree --output models_output/
"""

import argparse
import pickle
import logging
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.data import load_train_test
from src.models.decision_tree import DecisionTree
from src.models.naive_bayes import NaiveBayes

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def train_model(model_type, output_dir=None):
    """
    Train and save a model.
    
    Args:
        model_type: Type of model ('decision_tree', 'naive_bayes')
        output_dir: Directory to save model
        
    Returns:
        str: Path to saved model
    """
    if output_dir is None:
        output_dir = config.MODELS_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"📂 Output directory: {output_dir}")
    
    # Load data
    logger.info("📊 Loading data...")
    X_train, X_test, y_train, y_test = load_train_test(
        config.PREPROCESSED_DATA_PATH,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # Train model
    logger.info(f"🚀 Training {model_type} model...")
    
    if model_type == "decision_tree":
        model = DecisionTree(**config.MODELS_CONFIG['decision_tree'])
        model_name = "decision_tree"
    elif model_type == "naive_bayes":
        model = NaiveBayes(**config.MODELS_CONFIG['naive_bayes'])
        model_name = "naive_bayes"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"✅ Training complete!")
    
    # Evaluate
    from src.utils import accuracy_score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"   Test Accuracy: {acc:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"{model_name}_{timestamp}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"💾 Model saved: {model_path}")
    
    return str(model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train a model for heart disease prediction"
    )
    parser.add_argument(
        "--model",
        choices=["decision_tree", "naive_bayes"],
        default="decision_tree",
        help="Model type to train"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for model"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🫀 Heart Disease Prediction - Training Script")
    logger.info("=" * 60)
    
    model_path = train_model(args.model, args.output)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Training complete! Model saved to: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
