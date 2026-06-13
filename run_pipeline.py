import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.load_data import load_and_prepare_data
from src.models.train import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("TELCO CHURN PREDICTION PIPELINE")
    logger.info("=" * 60)
    logger.info("Step 1: Loading and preparing data...")
    X_train, X_test, y_train, y_test, cat_cols, num_cols = load_and_prepare_data()
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Categorical cols: {cat_cols}")
    logger.info(f"Numerical cols: {num_cols}")
    logger.info("Step 2: Training models...")
    best_pipeline, results = run_training_pipeline(X_train, y_train, X_test, y_test, cat_cols, num_cols)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
