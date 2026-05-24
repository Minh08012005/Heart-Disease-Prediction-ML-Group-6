import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành tập train và test.
    
    Parameters:
    - X: features (mảng 2D)
    - y: labels (mảng 1D)
    - test_size: tỷ lệ dữ liệu dùng để test (mặc định 0.2 = 20%)
    - random_state: seed để tái tạo kết quả
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    # Bước 1: Đặt seed để kết quả có thể tái tạo
    np.random.seed(random_state)
    
    # Bước 2: Lấy số lượng mẫu
    n_samples = len(X)
    
    # Bước 3: Tính số lượng mẫu cho test set
    n_test = int(n_samples * test_size)
    
    # Bước 4: Tạo mảng indices và trộn ngẫu nhiên
    indices = np.random.permutation(n_samples)
    
    # Bước 5: Lấy indices cho test và train
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Bước 6: Chia dữ liệu dựa trên indices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Bước 7: Trả về kết quả
    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_pred):
    """
    Tính độ chính xác (accuracy) của mô hình.
    
    Parameters:
    - y_true: nhãn thực tế (mảng 1D)
    - y_pred: nhãn dự đoán (mảng 1D)
    
    Returns:
    - accuracy: tỷ lệ dự đoán đúng (từ 0.0 đến 1.0)
    """
    return np.mean(y_true == y_pred)


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
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[TN, FP], [FN, TP]])


def precision_score(y_true, y_pred):
    """
    Tính Precision = TP / (TP + FP)
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - precision: từ 0.0 đến 1.0
    """
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    
    if (TP + FP) == 0:
        return 0.0
    
    return TP / (TP + FP)


def recall_score(y_true, y_pred):
    """
    Tính Recall = TP / (TP + FN)
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - recall: từ 0.0 đến 1.0
    """
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    
    if (TP + FN) == 0:
        return 0.0
    
    return TP / (TP + FN)


def f1_score(y_true, y_pred):
    """
    Tính F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - f1: từ 0.0 đến 1.0
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    if (p + r) == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


def k_fold_cross_validation(X, y, model_class, k=5, random_state=42, **model_params):
    """
    K-Fold Cross Validation
    
    Parameters:
    - X: features
    - y: labels
    - model_class: class của model (VD: DecisionTree)
    - k: số folds (mặc định 5)
    - random_state: seed
    - **model_params: tham số cho model (VD: max_depth=10, min_samples_split=20)
    
    Returns:
    - accuracies: list accuracy của từng fold
    """
    np.random.seed(random_state)
    n_samples = len(X)
    
    # 1. Trộn indices
    indices = np.random.permutation(n_samples)
    
    # 2. Tính kích thước mỗi fold
    fold_size = n_samples // k
    accuracies = []
    
    for i in range(k):
        # 3. Xác định indices cho test fold i
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n_samples
        
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        # 4. Chia dữ liệu
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 5. Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # 6. Predict và tính accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    return accuracies
