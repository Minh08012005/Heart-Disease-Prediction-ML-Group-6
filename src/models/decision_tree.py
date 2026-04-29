"""
decision_tree.py - Module thuật toán Decision Tree

Tác giả: Hiếu & Phong
Mô tả: Implement Decision Tree từ đầu dựa trên Entropy và Information Gain.
    
    Công thức:
    - Entropy: H(S) = -Σ pᵢ * log₂(pᵢ)
    - Information Gain: IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) * H(Sᵥ)
    
    Cây được xây dựng bằng đệ quy (recursive), chọn split tốt nhất
    dựa trên Information Gain lớn nhất.
"""
import numpy as np


class Node:
    """
    Cấu trúc một node trong Decision Tree.
    
    Attributes:
    - feature: chỉ số feature dùng để split (None nếu là leaf node)
    - threshold: ngưỡng split (None nếu là leaf node)
    - left: node con trái (<= threshold)
    - right: node con phải (> threshold)
    - value: giá trị dự đoán (chỉ có ở leaf node)
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """
    Decision Tree Classifier
    
    Parameters:
    - max_depth: độ sâu tối đa của cây (tránh overfitting)
    - min_samples_split: số mẫu tối thiểu để tiếp tục split
    
    Attributes:
    - root: Node gốc của cây
    - n_classes: số lượng class
    """
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None
    
    def _entropy(self, y):
        """
        Tính Entropy của tập labels.
        
        Công thức: H(S) = -Σ pᵢ * log₂(pᵢ)
        Trong đó pᵢ là tỷ lệ của class i trong tập S.
        
        Parameters:
        - y: numpy array, labels
        
        Returns:
        - entropy: giá trị entropy
        """
        # TODO: Implement from scratch
        # 1. Đếm số lượng mỗi class
        # 2. Tính tỷ lệ pᵢ
        # 3. Tính entropy = -Σ pᵢ * log₂(pᵢ)
        pass
    
    def _information_gain(self, X_column, y, threshold):
        """
        Tính Information Gain khi split tại threshold.
        
        Công thức: IG = H(parent) - (n_left/n)*H(left) - (n_right/n)*H(right)
        
        Parameters:
        - X_column: numpy array, giá trị của một feature
        - y: numpy array, labels
        - threshold: ngưỡng split
        
        Returns:
        - gain: information gain
        """
        # TODO: Implement from scratch
        # 1. Chia dữ liệu thành 2 phần dựa trên threshold
        # 2. Tính entropy của parent, left, right
        # 3. Tính information gain
        pass
    
    def _best_split(self, X, y):
        """
        Tìm split tốt nhất dựa trên Information Gain.
        
        Parameters:
        - X: numpy array, features
        - y: numpy array, labels
        
        Returns:
        - best_feature: chỉ số feature tốt nhất
        - best_threshold: ngưỡng split tốt nhất
        """
        # TODO: Implement from scratch
        # 1. Duyệt qua từng feature
        # 2. Với mỗi feature, thử các threshold khác nhau
        # 3. Chọn (feature, threshold) có IG lớn nhất
        pass
    
    def _build_tree(self, X, y, depth=0):
        """
        Xây dựng cây bằng đệ quy.
        
        Parameters:
        - X: numpy array, features
        - y: numpy array, labels
        - depth: độ sâu hiện tại
        
        Returns:
        - node: Node hiện tại
        """
        # TODO: Implement from scratch
        # 1. Kiểm tra điều kiện dừng (leaf node)
        # 2. Tìm split tốt nhất
        # 3. Chia dữ liệu và đệ quy xây cây con
        pass
    
    def fit(self, X, y):
        """
        Huấn luyện mô hình Decision Tree.
        
        Parameters:
        - X: numpy array, ma trận features (n_samples, n_features)
        - y: numpy array, nhãn (n_samples,)
        """
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
    
    def _traverse_tree(self, x, node):
        """
        Duyệt cây để dự đoán cho một sample.
        
        Parameters:
        - x: numpy array, một sample cần dự đoán
        - node: Node hiện tại
        
        Returns:
        - prediction: giá trị dự đoán
        """
        # TODO: Implement from scratch
        # 1. Nếu là leaf node, trả về value
        # 2. Nếu không, so sánh x[feature] với threshold
        # 3. Rẽ trái hoặc phải tùy theo kết quả
        pass
    
    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới.
        
        Parameters:
        - X: numpy array, ma trận features cần dự đoán
        
        Returns:
        - predictions: numpy array, nhãn dự đoán
        """
        # TODO: Implement from scratch
        pass
