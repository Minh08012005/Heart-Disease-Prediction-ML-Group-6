# PHẦN 2: CƠ SỞ LÝ THUYẾT

## 2.1 Entropy & Information Gain (Decision Tree)

### 2.1.1 Entropy

Entropy là đại lượng đo độ **hỗn loạn** (hoặc độ không chắc chắn) của một tập dữ liệu. Trong Decision Tree, Entropy được dùng để đánh giá mức độ "thuần khiết" của một tập labels.

**Công thức:**

$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Trong đó:

- $S$: tập dữ liệu
- $c$: số lượng class
- $p_i$: tỷ lệ của class $i$ trong tập $S$

**Ý nghĩa:**

- **H(S) = 0**: Tập dữ liệu hoàn toàn thuần khiết (tất cả cùng 1 class)
- **H(S) = 1**: Tập dữ liệu hỗn loạn nhất (50-50 giữa 2 class)

**Ví dụ:**

- Tập [0, 0, 0, 0]: H(S) = 0 (thuần khiết)
- Tập [0, 0, 1, 1]: H(S) = -(0.5×log₂(0.5) + 0.5×log₂(0.5)) = 1 (hỗn loạn nhất)
- Tập [0, 0, 0, 1]: H(S) = -(0.75×log₂(0.75) + 0.25×log₂(0.25)) = 0.81

### 2.1.2 Information Gain

Information Gain (IG) đo lường mức độ **giảm entropy** sau khi split dữ liệu theo một feature.

**Công thức:**

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \times H(S_v)$$

Trong đó:

- $S$: tập dữ liệu trước khi split
- $A$: feature dùng để split
- $S_v$: tập con sau khi split tại giá trị $v$
- $H(S_v)$: entropy của tập con

**Ý nghĩa:** IG càng lớn → feature đó càng tốt để split.

### 2.1.3 Thuật toán xây dựng cây

```
1. Bắt đầu với node gốc chứa toàn bộ dữ liệu
2. Với mỗi node:
   a. Nếu đạt điều kiện dừng (max_depth, min_samples_split, pure node) → tạo leaf node
   b. Tính Information Gain cho tất cả features
   c. Chọn feature có IG lớn nhất để split
   d. Chia dữ liệu thành 2 nhánh (≤ threshold và > threshold)
   e. Đệ quy xây dựng cây con cho mỗi nhánh
```

## 2.2 Bayes Theorem & Gaussian Naive Bayes

### 2.2.1 Bayes Theorem

Định lý Bayes mô tả xác suất của một sự kiện dựa trên các điều kiện liên quan:

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

Trong bài toán phân loại:

- $P(y|X)$: **Posterior** - xác suất sample thuộc class $y$ khi biết features $X$
- $P(X|y)$: **Likelihood** - xác suất có features $X$ nếu sample thuộc class $y$
- $P(y)$: **Prior** - xác suất tiên nghiệm của class $y$
- $P(X)$: **Evidence** - xác suất của features $X$

### 2.2.2 Giả định Naive

Naive Bayes giả định rằng các features **độc lập với nhau** khi biết class:

$$P(X|y) = P(x_1, x_2, ..., x_n|y) = P(x_1|y) \times P(x_2|y) \times ... \times P(x_n|y)$$

Giả định này "ngây thơ" (naive) vì trong thực tế các features thường có mối quan hệ với nhau, nhưng nó giúp đơn giản hóa tính toán rất nhiều.

### 2.2.3 Gaussian Probability Density Function

Với các features số, ta giả định chúng tuân theo phân phối chuẩn (Gaussian):

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \times \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

Trong đó:

- $\mu_y$: giá trị trung bình của feature $x_i$ trong class $y$
- $\sigma_y^2$: phương sai của feature $x_i$ trong class $y$

### 2.2.4 Thuật toán Naive Bayes

```
1. Huấn luyện (fit):
   a. Tính prior P(y) cho mỗi class: P(y) = count(y) / n_samples
   b. Tính mean μ và variance σ² cho mỗi feature trong mỗi class

2. Dự đoán (predict):
   a. Với mỗi sample, tính log-posterior cho mỗi class:
      log P(y|X) = log P(y) + Σ log P(x_i|y)
   b. Chọn class có log-posterior lớn nhất
   (Dùng log để tránh underflow khi nhân nhiều xác suất nhỏ)
```

## 2.3 Các metrics đánh giá

### 2.3.1 Confusion Matrix

Ma trận 2×2 so sánh dự đoán với thực tế:

```
              Dự đoán
            ┌─────┬─────┐
            │  0  │  1  │
     ┌───┬──┼─────┼─────┤
     │ 0 │  │ TN  │ FP  │
Thực ├───┼──┼─────┼─────┤
tế   │ 1 │  │ FN  │ TP  │
     └───┴──┴─────┴─────┘
```

- **TN (True Negative)**: Dự đoán không bệnh ✅ - đúng
- **FP (False Positive)**: Dự đoán có bệnh ❌ - sai (chẩn đoán nhầm)
- **FN (False Negative)**: Dự đoán không bệnh ❌ - sai (bỏ sót người bệnh) ⚠️
- **TP (True Positive)**: Dự đoán có bệnh ✅ - đúng

### 2.3.2 Accuracy

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Tỷ lệ dự đoán đúng trên tổng số dự đoán.

### 2.3.3 Precision

$$Precision = \frac{TP}{TP + FP}$$

"Trong số những người được dự đoán là CÓ BỆNH, có bao nhiêu người thực sự có bệnh?"

### 2.3.4 Recall

$$Recall = \frac{TP}{TP + FN}$$

"Trong số những người THỰC SỰ có bệnh, mô hình phát hiện được bao nhiêu?" — **Rất quan trọng trong y tế!**

### 2.3.5 F1-Score

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

Trung bình điều hòa giữa Precision và Recall, cân bằng giữa 2 chỉ số.

### 2.3.6 K-Fold Cross Validation

Chia dữ liệu thành **k phần bằng nhau**. Mỗi lần lấy 1 phần làm test, k-1 phần còn lại làm train. Lặp lại k lần. Kết quả là trung bình của k lần đánh giá.

**Ưu điểm:** Đánh giá model khách quan hơn, tránh may rủi do cách chia train/test.
