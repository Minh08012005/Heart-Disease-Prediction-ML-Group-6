'''
Dưới đây là **TASK LIST CẤP TỐC** cho tuần này (23/4 - 30/4) với 4 task chính, mỗi người 1 task, deadline rõ ràng:

---

# **⚡ SPRINT CẤP TỐC: 23/4 - 30/4**

## **📋 TASK 1: MINH - EDA & PREPROCESSING (DEADLINE: 26/4)**

````markdown
### TASK-01: EDA & Xử lý giá trị 0 vô lý - CẤP TỐC

**Assignee:** @Minh  
**Deadline:** 26/4/2026 (3 ngày)  
**Priority:** CRITICAL 🔴  
**Labels:** preprocessing, data-cleaning, eda  
**Status:** To Do

**🎯 MỤC TIÊU CẤP TỐC:**
Cung cấp dataset đã làm sạch CHO TEAM CODE THUẬT TOÁN NGAY.

**📝 CÔNG VIỆC CỤ THỂ (Làm NGAY hôm nay):**

1. **Tải dataset** (nếu chưa có):
   - Download heart.csv từ Kaggle
   - Lưu vào `data/heart.csv`

2. **EDA nhanh (2 giờ):**

   ```python
   # notebooks/01_EDA_quick.ipynb
   import pandas as pd
   df = pd.read_csv('data/heart.csv')
   print("Shape:", df.shape)
   print("Columns:", df.columns.tolist())
   print("HeartDisease distribution:", df['HeartDisease'].value_counts())

   # QUAN TRỌNG: Tìm giá trị 0 vô lý
   print("Cholesterol=0:", (df['Cholesterol']==0).sum())
   print("RestingBP=0:", (df['RestingBP']==0).sum())
   ```
````

3. **Xử lý giá trị 0 VÔ LÝ (2 giờ):**
   - Cholesterol=0 → thay bằng median của NHÓM TƯƠNG ỨNG (có bệnh/không bệnh)
   - RestingBP=0 → thay bằng median chung
   - Lưu thành `data/heart_no_zeros.csv`

4. **Tạo utils.py CƠ BẢN (1 giờ):**

   ```python
   # src/utils.py
   import numpy as np

   def train_test_split(X, y, test_size=0.2, random_state=42):
       # Implement đơn giản
       pass

   def accuracy_score(y_true, y_pred):
       return np.mean(y_true == y_pred)
   ```

**✅ DELIVERABLES (Phải có trước 26/4):**

- [ ] `data/heart.csv` (dataset gốc)
- [ ] `data/heart_no_zeros.csv` (đã xử lý giá trị 0)
- [ ] `src/utils.py` (hàm cơ bản)
- [ ] Short report: "Dataset ready for algorithm development"

**🔗 LINK TÀI LIỆU:**

- [Kaggle: Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- [Code mẫu xử lý giá trị 0](https://gist.github.com/...)

````

---

## **📋 TASK 2: TUÂN - NAIVE BAYES IMPLEMENTATION (DEADLINE: 30/4)**

```markdown
### TASK-02: Implement Gaussian Naive Bayes từ đầu

**Assignee:** @Tuân
**Deadline:** 30/4/2026 (7 ngày)
**Priority:** HIGH 🔴
**Labels:** algorithm, naive-bayes, from-scratch
**Status:** To Do

**🎯 MỤC TIÊU:**
Có class NaiveBayes hoạt động được, predict được trên data đơn giản.

**📝 CÔNG VIỆC CỤ THỂ:**

**Ngày 1-2 (23-24/4): Nghiên cứu toán học**
1. Hiểu Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)
2. Hiểu Gaussian Distribution PDF:
````

P(x|μ,σ²) = 1/√(2πσ²) \* exp(-(x-μ)²/(2σ²))

````
3. Hiểu Naive Bayes assumption: Features độc lập

**Ngày 3-4 (25-26/4): Code structure**
```python
# src/models/naive_bayes.py
import numpy as np

class NaiveBayes:
 def __init__(self):
     self.priors = {}      # P(y) cho mỗi class
     self.means = {}       # mean của mỗi feature cho mỗi class
     self.variances = {}   # variance của mỗi feature cho mỗi class

 def fit(self, X, y):
     # 1. Tính priors: P(y) = count(y) / n_samples
     # 2. Tính means & variances cho từng feature & class
     pass

 def _gaussian_pdf(self, x, mean, var):
     # Tính Gaussian Probability Density Function
     return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

 def predict(self, X):
     # Với mỗi sample, tính posterior cho mỗi class
     # Chọn class có posterior lớn nhất
     pass
````

**Ngày 5-7 (27-30/4): Testing & Debug**

1. Test với data đơn giản:
   ```python
   X = [[1, 2], [2, 3], [3, 4], [4, 5]]
   y = [0, 0, 1, 1]
   model = NaiveBayes()
   model.fit(X, y)
   predictions = model.predict([[2.5, 3.5]])
   ```
2. Test với dataset thật (sau khi Minh xử lý xong)

**✅ DELIVERABLES (30/4):**

- [ ] `src/models/naive_bayes.py` hoàn chỉnh
- [ ] Test script chứng minh hoạt động
- [ ] Accuracy > 70% trên test set đơn giản

**🔗 TÀI LIỆU THAM KHẢO:**

- [Bayes Theorem Explained](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [Gaussian Naive Bayes Math](https://www.geeksforgeeks.org/naive-bayes-classifiers/)
- [Code along tutorial](https://github.com/...)

````

---

## **📋 TASK 3: HIẾU & PHONG - DECISION TREE IMPLEMENTATION (DEADLINE: 30/4)**

```markdown
### TASK-03: Implement Decision Tree từ đầu (Pair Programming)

**Assignees:** @Hiếu, @Phong
**Deadline:** 30/4/2026 (7 ngày)
**Priority:** HIGH 🔴
**Labels:** algorithm, decision-tree, from-scratch, pair-programming
**Status:** To Do

**🎯 MỤC TIÊU:**
Có class DecisionTree build được cây đơn giản, predict được.

**📝 CÔNG VIỆC CỤ THỂ:**

**Phân công pair programming:**
- **Hiếu**: Node structure, tree building logic
- **Phong**: Entropy/Information Gain calculation, splitting logic

**Ngày 1-2 (23-24/4): Nghiên cứu toán học**
1. **Entropy**: H(S) = -Σ pᵢ log₂(pᵢ)
2. **Information Gain**: IG(S,A) = H(S) - Σ (|Sᵥ|/|S|) * H(Sᵥ)
3. **Tree building algorithm** (recursive)

**Ngày 3-4 (25-26/4): Code structure**
```python
# src/models/decision_tree.py
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature    # Chỉ số feature để split
        self.threshold = threshold # Ngưỡng split
        self.left = left          # Node con trái (<= threshold)
        self.right = right        # Node con phải (> threshold)
        self.value = value        # Giá trị nếu là leaf node

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def _entropy(self, y):
        # Tính entropy của labels y
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _information_gain(self, X_column, y, threshold):
        # Tính information gain khi split tại threshold
        left_indices = X_column <= threshold
        right_indices = X_column > threshold

        n = len(y)
        n_left, n_right = np.sum(left_indices), np.sum(right_indices)

        if n_left == 0 or n_right == 0:
            return 0

        # Tính entropy cho parent, left, right
        entropy_parent = self._entropy(y)
        entropy_left = self._entropy(y[left_indices])
        entropy_right = self._entropy(y[right_indices])

        # Information Gain
        gain = entropy_parent - (n_left/n)*entropy_left - (n_right/n)*entropy_right
        return gain

    def fit(self, X, y):
        # Xây dựng cây (recursive)
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        # Duyệt cây để predict
        pass
````

**Ngày 5-7 (27-30/4): Testing & Debug**

1. Test với data đơn giản (2 features)
2. Test với dataset thật
3. Visualize cây (optional)

**✅ DELIVERABLES (30/4):**

- [ ] `src/models/decision_tree.py` hoàn chỉnh
- [ ] Cây có thể build được với depth=3
- [ ] Predict được trên test data
- [ ] Accuracy > 65% trên test set đơn giản

**🔗 TÀI LIỆU THAM KHẢO:**

- [Decision Tree Math Explained](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [Entropy & Information Gain](https://www.geeksforgeeks.org/entropy-information-gain-machine-learning/)
- [Recursive Tree Building](https://github.com/...)

````

---

## **📋 TASK 4: ALL TEAM - WEEKLY SYNC & INTEGRATION (DEADLINE: 30/4)**

```markdown
### TASK-04: Weekly Sync & Integration Check

**Assignees:** @Minh, @Tuân, @Hiếu, @Phong
**Deadline:** 30/4/2026
**Priority:** MEDIUM 🟡
**Labels:** meeting, integration, review
**Status:** To Do

**📅 MEETING SCHEDULE:**
- **26/4 (Thứ 7) - 19:00**: Checkpoint 1 (Minh demo preprocessing xong)
- **30/4 (Thứ 4) - 19:00**: Final Review (Demo algorithms)

**🎯 MỤC TIÊU BUỔI 30/4:**
Tất cả code tích hợp được vào `main.py` và chạy được end-to-end.

**📝 CÔNG VIỆC CHUNG:**
1. **Minh**: Tạo `main.py` skeleton
   ```python
   # main.py
   from src.models.naive_bayes import NaiveBayes
   from src.models.decision_tree import DecisionTree
   from src.utils import train_test_split, accuracy_score

   # Load data
   # Preprocessing
   # Train/test split
   # Train models
   # Evaluate
````

2. **Tất cả**: Integration testing
   - Đảm bảo imports work
   - Đảm bảo API consistency (fit/predict)

**✅ DELIVERABLES (30/4):**

- [ ] `main.py` chạy được end-to-end
- [ ] Cả 2 models train được
- [ ] Có evaluation metrics cơ bản
- [ ] Meeting notes & action items cho tuần sau

**📊 SUCCESS CRITERIA:**

- Project chạy được: `python main.py`
- Không có lỗi import
- Có output metrics (accuracy)

````

---

## **📊 GITHUB WORKFLOW CẤP TỐC:**

### **HÔM NAY (23/4) - BẠN LÀM NGAY:**
1. **Tạo 4 Issues** trên GitHub với nội dung trên
2. **Assign đúng người**
3. **Tạo branches:**
   ```bash
   git checkout -b minh/preprocessing
   git checkout -b tuan/naive-bayes
   git checkout -b hieu-phong/decision-tree
````

4. **Gửi task cho team qua chat group**

### **DAILY CHECK-IN (Từ 24/4):**

- **9:00 sáng**: Mỗi người update progress trên Issue
- **Blockers**: Tag @Minh nếu cần help

### **DEADLINE RÕ RÀNG:**

- **26/4**: Minh xong preprocessing → team có data để test
- **30/4**: Tất cả xong code → integration meeting

---

## **🚨 CẤP TỐC SUPPORT TÔI CÓ THỂ CUNG CẤP:**

**Option 1: Template code NGAY BÂY GIỜ**

- Tôi tạo file `src/models/naive_bayes_template.py` với structure đúng
- Tôi tạo file `src/models/decision_tree_template.py` với structure đúng
- Bạn chia cho team để họ fill logic

**Option 2: Toán học giải thích chi tiết**

- Tôi giải thích từng công thức với ví dụ số cụ thể
- Tạo cheat sheet toán học cho từng thuật toán

**Option 3: Test data đơn giản**

- Tôi tạo file `test_simple_data.py` với data nhỏ để test
- Đảm bảo team có thể test code ngay

**Option 4: Tất cả trên**

**Bạn muốn tôi bắt đầu với Option nào?** Tôi có thể làm NGAY BÂY GIỜ để team bắt đầu code ngay trong hôm nay.
