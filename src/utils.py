"""
utils.py - Các hàm tiện ích dùng chung cho toàn bộ dự án

Các hàm trong file này được sử dụng bởi tất cả các module khác.
Mỗi hàm đều được code từ đầu (from scratch) để phục vụ mục đích học tập.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành tập train và test.
    
    Parameters:
    - X: features
    - y: labels
    - test_size: tỷ lệ dữ liệu dùng để test (mặc định 20%)
    - random_state: seed để tái tạo kết quả
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = len(y)
    n_test = int(n_samples * test_size)

    # Tao index shuffle de dam bao random nhung reproducible
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def accuracy_score(y_true, y_pred):
    """
    Tính độ chính xác (accuracy) của mô hình.
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - accuracy: tỷ lệ dự đoán đúng
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """
    Tạo ma trận nhầm lẫn (Confusion Matrix).
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - Ma trận 2x2: [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])


def precision_score(y_true, y_pred):
    """
    Tính Precision = TP / (TP + FP)
    Trong y tế: Trong số những người được dự đoán có bệnh, bao nhiêu người thực sự có bệnh?
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall_score(y_true, y_pred):
    """
    Tính Recall = TP / (TP + FN)
    Trong y tế: Trong số những người thực sự có bệnh, mô hình phát hiện được bao nhiêu?
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def f1_score(y_true, y_pred):
    """
    Tính F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    Là trung bình điều hòa giữa Precision và Recall.
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)
