"""
utils.py - Các hàm tiện ích dùng chung cho toàn bộ dự án

Các hàm trong file này được sử dụng bởi tất cả các module khác.
Mỗi hàm đều được code từ đầu (from scratch) để phục vụ mục đích học tập.
"""




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
    import random
    X = list(X)
    y = list(y)
    n = len(X)
    if isinstance(test_size, float):
        test_n = int(n * test_size)
    else:
        test_n = int(test_size)

    rng = random.Random(random_state)
    indices = list(range(n))
    rng.shuffle(indices)

    test_idx = set(indices[:test_n])
    X_train = [X[i] for i in range(n) if i not in test_idx]
    X_test = [X[i] for i in range(n) if i in test_idx]
    y_train = [y[i] for i in range(n) if i not in test_idx]
    y_test = [y[i] for i in range(n) if i in test_idx]

    # convert to numpy arrays
    import numpy as np
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def accuracy_score(y_true, y_pred):
    """
    Tính độ chính xác (accuracy) của mô hình.
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - accuracy: tỷ lệ dự đoán đúng
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if len(y_true) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """
    Tạo ma trận nhầm lẫn (Confusion Matrix).
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - Ma trận 2x2: [[TN, FP], [FN, TP]]
    """
    # Assuming binary labels 0 and 1
    tn = fp = fn = tp = 0
    for a, b in zip(y_true, y_pred):
        if a == 0 and b == 0:
            tn += 1
        elif a == 0 and b == 1:
            fp += 1
        elif a == 1 and b == 0:
            fn += 1
        elif a == 1 and b == 1:
            tp += 1
    return [[tn, fp], [fn, tp]]


def precision_score(y_true, y_pred):
    """
    Tính Precision = TP / (TP + FP)
    Trong y tế: Trong số những người được dự đoán có bệnh, bao nhiêu người thực sự có bệnh?
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fp = cm[0][1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    """
    Tính Recall = TP / (TP + FN)
    Trong y tế: Trong số những người thực sự có bệnh, mô hình phát hiện được bao nhiêu?
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fn = cm[1][0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """
    Tính F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    Là trung bình điều hòa giữa Precision và Recall.
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
