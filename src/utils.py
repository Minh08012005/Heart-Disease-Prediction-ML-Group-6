import numpy as np





def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Chia dữ liệu thành tập train và test.
    
    Parameters:
    - X: features (mảng 2D, ví dụ: [[age, chol], [age, chol], ...])
    - y: labels (mảng 1D, ví dụ: [0, 1, 0, 1, ...])
    - test_size: tỷ lệ dữ liệu dùng để test (mặc định 0.2 = 20%)
    - random_state: seed để tái tạo kết quả
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    # Bước 1: Đặt seed để kết quả có thể tái tạo
    np.random.seed(random_state)
    
    # Bước 2: Lấy số lượng mẫu
    n_samples = len(X)  # Ví dụ: 918 samples
    
    # Bước 3: Tính số lượng mẫu cho test set
    n_test = int(n_samples * test_size)  # 918 * 0.2 = 183.6 → 183 samples
    
    # Bước 4: Tạo mảng indices [0, 1, 2, ..., 917] và trộn ngẫu nhiên
    indices = np.random.permutation(n_samples)
    # Ví dụ kết quả: [342, 12, 567, 89, ..., 901]
    
    # Bước 5: Lấy indices cho test (183 phần tử đầu) và train (phần còn lại)
    test_indices = indices[:n_test]    # 183 indices đầu
    train_indices = indices[n_test:]   # 918 - 183 = 735 indices còn lại
    
    # Bước 6: Chia dữ liệu dựa trên indices
    X_train = X[train_indices]  # Lấy các dòng có index trong train_indices
    X_test = X[test_indices]    # Lấy các dòng có index trong test_indices
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Bước 7: Trả về kết quả
    return X_train, X_test, y_train, y_test

"""
## Bước 2: Implement `accuracy_score` 📊

### Giải thích toán học

__Accuracy__ = (Số dự đoán đúng) / (Tổng số dự đoán)

Ví dụ:

- y_true = [0, 1, 0, 1, 0] (thực tế)
- y_pred = [0, 1, 1, 1, 0] (dự đoán)
- So sánh: [✅, ✅, ❌, ✅, ✅] → 4/5 đúng → accuracy = 0.8

"""
def accuracy_score(y_true, y_pred):
    """
    Tính độ chính xác (accuracy) của mô hình.
    
    Parameters:
    - y_true: nhãn thực tế (mảng 1D)
    - y_pred: nhãn dự đoán (mảng 1D)
    
    Returns:
    - accuracy: tỷ lệ dự đoán đúng (từ 0.0 đến 1.0)
    
    Ví dụ:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    accuracy = 2/4 = 0.5
    """
    # y_true == y_pred tạo ra mảng boolean: [True, True, False, False]
    # np.mean đếm số True / tổng số = 2/4 = 0.5
    return np.mean(y_true == y_pred)

## Bước 3: Implement `confusion_matrix` 📋
"""""
### Giải thích toán học

Confusion Matrix là bảng 2x2 so sánh dự đoán với thực tế:

```javascript
              Dự đoán
            ┌─────┬─────┐
            │  0  │  1  │
     ┌───┬──┼─────┼─────┤
     │ 0 │  │ TN  │ FP  │
Thực ├───┼──┼─────┼─────┤
tế   │ 1 │  │ FN  │ TP  │
     └───┴──┴─────┴─────┘
```

__Ý nghĩa trong bài toán bệnh tim:__

- __TN (True Negative)__: Dự đoán không bệnh ✅ — đúng, người khỏe
- __FP (False Positive)__: Dự đoán có bệnh ❌ — sai, người khỏe bị chẩn đoán nhầm
- __FN (False Negative)__: Dự đoán không bệnh ❌ — sai, __nguy hiểm nhất!__ (bỏ sót người bệnh)
- __TP (True Positive)__: Dự đoán có bệnh ✅ — đúng, phát hiện được người bệnh
"""
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


"""
## Bước 4: Implement `precision_score` 🎯

### Giải thích toán học

__Precision__ = TP / (TP + FP)

__Ý nghĩa trong y tế:__

> "Trong số những người được mô hình dự đoán là CÓ BỆNH, có bao nhiêu người thực sự có bệnh?"

- Precision cao → ít bị chẩn đoán nhầm (FP thấp)
- Precision thấp → nhiều người khỏe bị chẩn đoán nhầm là có bệnh

"""
def precision_score(y_true, y_pred):
    """
    Tính Precision = TP / (TP + FP)
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - precision: từ 0.0 đến 1.0
    
    Ví dụ:
    CM = [[2, 1],    → TP=2, FP=1
          [1, 2]]
    precision = 2 / (2+1) = 0.667
    """
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  # Hàng 1, cột 1
    FP = cm[0, 1]  # Hàng 0, cột 1
    
    # Tránh chia cho 0
    if (TP + FP) == 0:
        return 0.0
    
    return TP / (TP + FP)

"""
## Bước 5: Implement `recall_score` 🎯

### Giải thích toán học

__Recall__ = TP / (TP + FN)

__Ý nghĩa trong y tế:__

> "Trong số những người THỰC SỰ có bệnh, mô hình phát hiện được bao nhiêu?"

- Recall cao → ít bỏ sót người bệnh (FN thấp) — __rất quan trọng trong y tế!__
- Recall thấp → bỏ sót nhiều người bệnh → nguy hiểm

"""
def recall_score(y_true, y_pred):
    """
    Tính Recall = TP / (TP + FN)
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - recall: từ 0.0 đến 1.0
    
    Ví dụ:
    CM = [[2, 1],    → TP=2, FN=1
          [1, 2]]
    recall = 2 / (2+1) = 0.667
    """
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  # Hàng 1, cột 1
    FN = cm[1, 0]  # Hàng 1, cột 0
    
    # Tránh chia cho 0
    if (TP + FN) == 0:
        return 0.0
    
    return TP / (TP + FN)

"""
## Bước 6: Implement `f1_score` 🎯

### Giải thích toán học

__F1-Score__ = 2 × (Precision × Recall) / (Precision + Recall)

__Tại sao cần F1-Score?__

- Precision và Recall thường __đối nghịch__ nhau:

  - Precision cao → Recall thấp (chỉ dự đoán khi chắc chắn → bỏ sót nhiều)
  - Recall cao → Precision thấp (dự đoán thoáng → nhiều FP)

- F1-Score là __trung bình điều hòa__ giữa 2 cái → cân bằng hơn

"""
def f1_score(y_true, y_pred):
    """
    Tính F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - f1: từ 0.0 đến 1.0
    
    Ví dụ:
    precision = 0.667, recall = 0.667
    f1 = 2 * (0.667 * 0.667) / (0.667 + 0.667) = 0.667
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    # Tránh chia cho 0
    if (p + r) == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)
