# PHẦN 5: THUẬT TOÁN TỰ CODE

## 5.1 Decision Tree (Hiếu & Phong + Minh)

### 5.1.1 Cấu trúc Node

Mỗi node trong Decision Tree được biểu diễn bằng class `Node`:

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Chỉ số feature để split
        self.threshold = threshold  # Ngưỡng split
        self.left = left            # Node con trái (<= threshold)
        self.right = right          # Node con phải (> threshold)
        self.value = value          # Giá trị nếu là leaf node
```

- **Leaf node**: node có `value != None` - chứa kết quả dự đoán
- **Internal node**: node có `feature` và `threshold` - dùng để split dữ liệu

### 5.1.2 Entropy

Tính độ hỗn loạn của tập labels:

```python
def _entropy(self, y):
    counts = np.bincount(y)
    proportions = counts / len(y)
    return -sum(p * np.log2(p) for p in proportions if p > 0)
```

**Ví dụ:**

- y = [0, 0, 1, 1]: proportions = [0.5, 0.5] → entropy = -(0.5×log₂0.5 + 0.5×log₂0.5) = 1.0
- y = [0, 0, 0, 0]: proportions = [1.0] → entropy = -(1.0×log₂1.0) = 0.0

### 5.1.3 Information Gain

Tính mức độ giảm entropy khi split:

```python
def _information_gain(self, X_column, y, threshold):
    left_mask = X_column <= threshold
    right_mask = ~left_mask
    n = len(y)
    n_left, n_right = np.sum(left_mask), np.sum(right_mask)

    if n_left == 0 or n_right == 0:
        return 0.0

    h_parent = self._entropy(y)
    h_left = self._entropy(y[left_mask])
    h_right = self._entropy(y[right_mask])

    gain = h_parent - (n_left / n) * h_left - (n_right / n) * h_right
    return gain
```

### 5.1.4 Tìm split tốt nhất

Duyệt qua tất cả features và thresholds để tìm (feature, threshold) có IG lớn nhất:

```python
def _best_split(self, X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature_idx in range(X.shape[1]):
        X_col = X[:, feature_idx]
        thresholds = np.unique(X_col)
        for threshold in thresholds:
            gain = self._information_gain(X_col, y, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold
```

### 5.1.5 Xây dựng cây (đệ quy)

```python
def _build_tree(self, X, y, depth=0):
    n_samples = len(y)
    n_classes = len(np.unique(y))

    # Điều kiện dừng
    if (self.max_depth is not None and depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_classes == 1):
        return Node(value=self._most_common_label(y))

    # Tìm split tốt nhất
    best_feature, best_threshold = self._best_split(X, y)

    if best_feature is None:
        return Node(value=self._most_common_label(y))

    # Chia dữ liệu và đệ quy
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
    right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

    return Node(feature=best_feature, threshold=best_threshold,
                left=left_child, right=right_child)
```

**Điều kiện dừng (stop conditions):**

1. `depth >= max_depth`: Đạt độ sâu tối đa
2. `n_samples < min_samples_split`: Quá ít mẫu để split tiếp
3. `n_classes == 1`: Node đã thuần khiết (tất cả cùng 1 class)

### 5.1.6 Dự đoán

Duyệt cây từ gốc đến lá:

```python
def _traverse_tree(self, x, node):
    if node.is_leaf:
        return node.value
    if x[node.feature] <= node.threshold:
        return self._traverse_tree(x, node.left)
    else:
        return self._traverse_tree(x, node.right)

def predict(self, X):
    return np.array([self._traverse_tree(x, self.root) for x in X])
```

### 5.1.7 Kết quả thực nghiệm

**Kết quả tuning tham số (7 độ sâu & split thresholds):**

Sau khi thử nghiệm với các tham số khác nhau trên train set, ta thu được:

| max_depth | min_samples_split |  Accuracy  | Precision  | Recall |  F1-Score  |
| :-------: | :---------------: | :--------: | :--------: | :----: | :--------: |
|     3     |         2         |   79.78%   |   86.32%   | 77.36% |   81.59%   |
|     5     |         2         | **82.51%** | **89.36%** | 79.25% | **84.00%** |
|     5     |        10         |   81.97%   |   88.42%   | 79.25% |   83.58%   |
|    10     |         2         |   80.33%   |   89.77%   | 74.53% |   81.44%   |
|    10     |        20         |   79.78%   |   87.91%   | 75.47% |   81.22%   |
|    15     |        20         |   79.78%   |   87.91%   | 75.47% |   81.22%   |
|    20     |        50         |   78.14%   |   84.38%   | 76.42% |   80.20%   |

**Best params:** max_depth=5, min_samples_split=2

- **Accuracy: 82.51%** (cao nhất)
- **Precision: 89.36%** (cao nhất)
- **Recall: 79.25%**
- **F1-Score: 84.00%** (cao nhất)

**Nhận xét:** Tham số tối ưu hóa cho F1-Score là max_depth=5, min_samples_split=2, đạt 82.51% Accuracy. Tuy nhiên, khi sử dụng max_depth=10, min_samples_split=20 để kiểm soát complexity và tránh overfitting, kết quả vẫn chấp nhận được (79.78% Accuracy, 81.22% F1-Score).

**K-Fold Cross Validation (k=5):**

- Mean Accuracy: 81.94%
- Standard Deviation: 3.94%

## 5.2 Naive Bayes (Tuân)

### 5.2.1 Cấu trúc class

```python
class NaiveBayes:
    def __init__(self):
        self.priors = {}      # P(y) cho mỗi class
        self.means = {}       # mean của mỗi feature cho mỗi class
        self.variances = {}   # variance của mỗi feature cho mỗi class
        self.classes = None
```

### 5.2.2 Huấn luyện (fit)

Tính prior, mean và variance cho từng class:

```python
def fit(self, X, y):
    self.classes = np.unique(y)
    n_samples = len(X)

    for c in self.classes:
        X_c = X[y == c]
        # Prior: P(y) = count(y) / n_samples
        self.priors[c] = len(X_c) / n_samples
        # Mean của mỗi feature trong class c
        self.means[c] = np.mean(X_c, axis=0)
        # Variance của mỗi feature trong class c
        self.variances[c] = np.var(X_c, axis=0)
```

### 5.2.3 Gaussian PDF

Tính xác suất của một giá trị trong phân phối Gaussian:

```python
def _gaussian_pdf(self, x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))
```

### 5.2.4 Dự đoán

Sử dụng log-posterior để tránh underflow:

```python
def predict(self, X):
    predictions = []
    for x in X:
        posteriors = []
        for c in self.classes:
            # log prior
            log_prior = np.log(self.priors[c])
            # log likelihood: Σ log P(x_i | y)
            log_likelihood = np.sum(np.log(
                self._gaussian_pdf(x, self.means[c], self.variances[c]) + 1e-9
            ))
            # log posterior = log prior + log likelihood
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        # Chọn class có posterior lớn nhất
        predictions.append(self.classes[np.argmax(posteriors)])

    return np.array(predictions)
```

**Tại sao dùng log?**

- Khi nhân nhiều xác suất nhỏ (ví dụ: 0.3 × 0.4 × 0.2 × ...), kết quả có thể dưới 10⁻³⁰⁸ (underflow)
- Dùng log: log(a×b) = log(a) + log(b) → cộng các số âm → không bị underflow
- log là hàm đơn điệu tăng → class nào có posterior lớn nhất thì log-posterior cũng lớn nhất

### 5.2.5 Kết quả thực nghiệm

_(Chờ cập nhật kết quả từ notebook của Tuân)_
