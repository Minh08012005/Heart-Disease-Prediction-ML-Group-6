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
        # Đếm số lượng mỗi class
        counts = np.bincount(y.astype(int))
        # Tính tỷ lệ pᵢ (bỏ qua class có count = 0)
        proportions = counts / len(y)
        # Tính entropy = -Σ pᵢ * log₂(pᵢ)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
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
        # Chia dữ liệu thành 2 phần dựa trên threshold
        left_mask = X_column <= threshold
        right_mask = X_column > threshold
        
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        # Nếu một bên rỗng thì gain = 0
        if n_left == 0 or n_right == 0:
            return 0
        
        # Tính entropy của parent, left, right
        entropy_parent = self._entropy(y)
        entropy_left = self._entropy(y[left_mask])
        entropy_right = self._entropy(y[right_mask])
        
        # Information Gain
        gain = entropy_parent - (n_left / n) * entropy_left - (n_right / n) * entropy_right
        return gain
    
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
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Duyệt qua từng feature
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            # Lấy các giá trị unique làm threshold candidates
            thresholds = np.unique(X_column)
            
            # Với mỗi threshold, tính Information Gain
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                
                # Cập nhật best nếu gain lớn hơn
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
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
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Điều kiện dừng → tạo leaf node
        # 1. Đã đạt max_depth
        # 2. Chỉ còn 1 class (pure node)
        # 3. Số mẫu < min_samples_split
        if (depth >= self.max_depth or
                n_classes == 1 or
                n_samples < self.min_samples_split):
            # Leaf node: trả về class phổ biến nhất
            leaf_value = np.bincount(y.astype(int)).argmax()
            return Node(value=leaf_value)
        
        # Tìm split tốt nhất
        best_feature, best_threshold = self._best_split(X, y)
        
        # Nếu không tìm được split tốt → leaf node
        if best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return Node(value=leaf_value)
        
        # Chia dữ liệu và đệ quy xây cây con
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        """
        Huấn luyện mô hình Decision Tree.
        
        Parameters:
        - X: numpy array, ma trận features (n_samples, n_features)
        - y: numpy array, nhãn (n_samples,)
        """
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(np.array(X), np.array(y))
    
    def _traverse_tree(self, x, node):
        """
        Duyệt cây để dự đoán cho một sample.
        
        Parameters:
        - x: numpy array, một sample cần dự đoán
        - node: Node hiện tại
        
        Returns:
        - prediction: giá trị dự đoán
        """
        # Nếu là leaf node, trả về value
        if node.value is not None:
            return node.value
        
        # So sánh x[feature] với threshold để rẽ trái hoặc phải
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới.
        
        Parameters:
        - X: numpy array, ma trận features cần dự đoán
        
        Returns:
        - predictions: numpy array, nhãn dự đoán
        """
        X = np.array(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
