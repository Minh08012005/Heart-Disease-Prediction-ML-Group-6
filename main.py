"""
main.py - File chạy chính của dự án Heart Disease Prediction

Quy trình:
1. Nạp dữ liệu từ data/heart_cleaned.csv
2. Tiền xử lý (nếu cần)
3. Chia train/test
4. Huấn luyện các mô hình (Naive Bayes, Decision Tree)
5. Đánh giá và so sánh kết quả
"""

import numpy as np
import pandas as pd

from src.utils import train_test_split, accuracy_score, confusion_matrix
from src.models.naive_bayes import NaiveBayes
from src.models.decision_tree import DecisionTree


def main():
    """
    Hàm chính thực hiện toàn bộ pipeline.
    """
    print("=" * 60)
    print("🫀 HEART DISEASE PREDICTION - ML PROJECT")
    print("=" * 60)
    
    # TODO: Implement full pipeline
    # 1. Load data
    # 2. Preprocessing
    # 3. Train/test split
    # 4. Train models
    # 5. Evaluate
    # 6. Compare results
    
    print("\n✅ Project structure is ready!")
    print("📝 Please implement the pipeline step by step.")


if __name__ == "__main__":
    main()
