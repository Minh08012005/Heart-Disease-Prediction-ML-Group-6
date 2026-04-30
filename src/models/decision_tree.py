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
    
    @property
    def is_leaf(self):
        """Kiem tra node co phai la leaf node khong."""
        return self.value is not None
    
    def __repr__(self):
        """Hien thi thong tin node khi debug."""
        if self.is_leaf:
            return f"Node(leaf, value={self.value})"
        return f"Node(feature={self.feature}, threshold={self.threshold})"


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
        self.n_features_ = None  # Luu so features khi fit, dung de validate predict
    
    def _most_common_label(self, y):
        """
        Tra ve nhan xuat hien nhieu nhat trong y.
        Dung de tao leaf node.
        """
        return int(np.bincount(y).argmax())
    
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
        # 1. Đếm số lượng mỗi class
        counts = np.bincount(y)
        # 2. Tính tỷ lệ pᵢ (bỏ qua class không xuất hiện)
        proportions = counts / len(y)
        # 3. Tính entropy = -Σ pᵢ * log₂(pᵢ)
        return -sum(p * np.log2(p) for p in proportions if p > 0)
    
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
        # 1. Chia dữ liệu thành 2 phần dựa trên threshold
        left_mask  = X_column <= threshold
        right_mask = ~left_mask
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)

        # Nếu một nhánh rỗng thì không split được
        if n_left == 0 or n_right == 0:
            return 0.0

        # 2. Tính entropy của parent, left, right
        h_parent = self._entropy(y)
        h_left   = self._entropy(y[left_mask])
        h_right  = self._entropy(y[right_mask])

        # 3. Tính information gain
        gain = h_parent - (n_left / n) * h_left - (n_right / n) * h_right
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
        # 1. Duyệt qua từng feature
        best_gain      = -1
        best_feature   = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            X_col = X[:, feature_idx]
            # 2. Với mỗi feature, thử các threshold là mỗi giá trị duy nhất
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._information_gain(X_col, y, threshold)
                # 3. Chọn (feature, threshold) có IG lớn nhất
                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """
        Xay dung cay bang de quy.
        
        Parameters:
        - X: numpy array, features
        - y: numpy array, labels
        - depth: do sau hien tai
        
        Returns:
        - node: Node hien tai
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # 1. Kiem tra dieu kien dung -> tao leaf node
        #    a. Dat max_depth
        #    b. Qua it mau de tiep tuc split
        #    c. Tat ca nhan giong nhau (pure node)
        if (depth >= self.max_depth
                or n_samples < self.min_samples_split
                or n_classes == 1):
            return Node(value=self._most_common_label(y))

        # 2. Tim split tot nhat
        best_feature, best_threshold = self._best_split(X, y)

        # Khong tim duoc split co ich
        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # 3. Chia du lieu va de quy xay cay con
        left_mask  = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # [FIX] Bao ve truong hop split ra 1 nhanh rong
        # Xay ra khi tat ca gia tri cua feature giong nhau
        # -> threshold = gia tri do -> moi mau di trai, phai rong -> crash
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return Node(value=self._most_common_label(y))

        left_child  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature   = best_feature,
            threshold = best_threshold,
            left      = left_child,
            right     = right_child,
        )
    
    def fit(self, X, y):
        """
        Huấn luyện mô hình Decision Tree.
        
        Parameters:
        - X: numpy array, ma trận features (n_samples, n_features)
        - y: numpy array, nhãn (n_samples,)
        
        Raises:
        - ValueError: nếu X và y có số lượng mẫu khác nhau
        """
        X = np.array(X)
        y = np.array(y, dtype=int)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X co {X.shape[0]} mau nhung y co {y.shape[0]} mau."
            )
        
        self.n_features_ = X.shape[1]
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
        # 1. Nếu là leaf node, trả về value
        if node.is_leaf:
            return node.value

        # 2. So sánh x[feature] với threshold
        # 3. Rẽ trái hoặc phải tùy theo kết quả
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
        
        Raises:
        - RuntimeError: nếu chưa gọi fit()
        """
        if self.root is None:
            raise RuntimeError("Chua huan luyen! Hay goi fit() truoc khi predict().")
        
        X = np.array(X)
        
        # Validate so features phai khop voi luc fit
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Fit voi {self.n_features_} features nhung predict voi {X.shape[1]} features."
            )
        
        return np.array([self._traverse_tree(x, self.root) for x in X])
    