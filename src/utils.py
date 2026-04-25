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
    # TODO: Implement from scratch
    pass


def accuracy_score(y_true, y_pred):
    """
    Tính độ chính xác (accuracy) của mô hình.
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - accuracy: tỷ lệ dự đoán đúng
    """
    # TODO: Implement from scratch
    pass


def confusion_matrix(y_true, y_pred):
    """
    Tạo ma trận nhầm lẫn (Confusion Matrix).
    
    Parameters:
    - y_true: nhãn thực tế
    - y_pred: nhãn dự đoán
    
    Returns:
    - Ma trận 2x2: [[TN, FP], [FN, TP]]
    """
    # TODO: Implement from scratch
    pass


def precision_score(y_true, y_pred):
    """
    Tính Precision = TP / (TP + FP)
    Trong y tế: Trong số những người được dự đoán có bệnh, bao nhiêu người thực sự có bệnh?
    """
    # TODO: Implement from scratch
    pass


def recall_score(y_true, y_pred):
    """
    Tính Recall = TP / (TP + FN)
    Trong y tế: Trong số những người thực sự có bệnh, mô hình phát hiện được bao nhiêu?
    """
    # TODO: Implement from scratch
    pass


def f1_score(y_true, y_pred):
    """
    Tính F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    Là trung bình điều hòa giữa Precision và Recall.
    """
    # TODO: Implement from scratch
    pass
