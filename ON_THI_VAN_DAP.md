# 📚 ÔN THI VẤN ĐÁP - HEART DISEASE PREDICTION

## Đi theo từng Cell trong Colab Notebook

---

# 🟢 CELL 1: IMPORT THƯ VIỆN & CÀI ĐẶT

## Lý thuyết cần nhớ: **Bài toán Machine Learning cơ bản**

### 🔹 Phân loại bài toán:

- **Supervised Learning** (học có giám sát): vì có nhãn `HeartDisease` (0/1)
- **Classification** (phân loại): vì đầu ra là 2 lớp rời rạc
- **Binary Classification** (phân loại nhị phân): 2 lớp (có bệnh / không bệnh)
- **Tabular Data** (dữ liệu dạng bảng): 918 hàng × 12 cột

### Câu hỏi cô có thể hỏi:

> **Q:** "Đây là bài toán supervised learning hay unsupervised learning? Tại sao?"
> **A:** "Supervised learning vì chúng em có biến mục tiêu HeartDisease (nhãn) để model học. Nếu không có nhãn thì mới là unsupervised."

> **Q:** "Bài toán regression hay classification?"
> **A:** "Classification ạ. Vì đầu ra là 2 giá trị rời rạc 0 và 1 (có bệnh/không bệnh), không phải giá trị liên tục như regression."

> **Q:** "Thế nào là Binary Classification?"
> **A:** "Là bài toán phân loại chỉ có 2 lớp. Ví dụ: có bệnh/không bệnh, spam/not spam, nam/nữ."

---

# 🟢 CELL 2: THÔNG TIN CƠ BẢN (head, info, describe)

## Lý thuyết: **EDA (Exploratory Data Analysis)**

### 🔹 EDA là gì?

- Quá trình khám phá dữ liệu để hiểu cấu trúc, phát hiện bất thường
- Dùng thống kê mô tả (describe) và trực quan hóa
- Mục tiêu: hiểu dữ liệu trước khi xây dựng model

### 🔹 Các hàm quan trọng:

- `head()`: xem 5 dòng đầu → kiểm tra dữ liệu đã load đúng chưa
- `info()`: xem kiểu dữ liệu, số lượng null → phát hiện missing values
- `describe()`: thống kê (mean, std, min, max, quartiles) → hiểu phân bố

### Câu hỏi cô có thể hỏi:

> **Q:** "Tại sao phải làm EDA?"
> **A:** "Để hiểu dữ liệu trước khi build model. Phát hiện missing values, outliers, mối quan hệ giữa features. Nếu bỏ qua EDA có thể dẫn đến model sai."

> **Q:** "Describe() cho em biết những thông tin gì?"
> **A:** "Cho biết count, mean, std, min, 25%, 50%, 75%, max của từng cột số. Giúp em biết được phân bố dữ liệu."

---

# 🟢 CELL 3: GIẢI THÍCH FEATURES & PHÂN BỐ TARGET

## Lý thuyết: **Feature & Target**

### 🔹 Feature (đặc trưng/đầu vào):

- 11 features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

### 🔹 Target (mục tiêu/đầu ra):

- HeartDisease: 0 = không bệnh, 1 = có bệnh

### 🔹 Phân bố target:

- 410 người không bệnh (44.7%)
- 508 người có bệnh (55.3%)
- → Gần cân bằng (không bị imbalanced)

### Câu hỏi cô có thể hỏi:

> **Q:** "Số lượng 2 class có mất cân bằng không?"
> **A:** "Không ạ. Class 0 có 410 mẫu (44.7%), class 1 có 508 mẫu (55.3%). Gần như cân bằng, không cần xử lý thêm."

> **Q:** "Imbalanced data là gì? Khi nào cần xử lý?"
> **A:** "Imbalanced là khi 1 class chiếm đa số (VD: 90% không bệnh, 10% có bệnh). Khi đó model có thể học tủ - chỉ đoán class đa số. Cần xử lý bằng SMOTE, oversampling, hoặc dùng weighted loss."

---

# 🟢 CELL 4: PHÁT HIỆN & XỬ LÝ GIÁ TRỊ 0 VÔ LÝ

## Lý thuyết: **Data Cleaning**

### 🔹 Vấn đề:

- **Cholesterol**: 172 giá trị = 0 (vô lý vì cholesterol không thể = 0)
- **RestingBP**: 1 giá trị = 0 (vô lý)

### 🔹 Cách xử lý:

- **Imputation**: thay thế bằng median theo từng nhóm (bệnh/không bệnh)
- Tại sao dùng median? → robust với outliers hơn mean
- Tại sao theo nhóm? → vì người có bệnh thường có cholesterol khác người không bệnh

### Câu hỏi cô có thể hỏi:

> **Q:** "Tại sao Cholesterol = 0 là vô lý?"
> **A:** "Về mặt y học, cholesterol không thể bằng 0. Đây là lỗi nhập liệu hoặc missing value được ghi là 0."

> **Q:** "Tại sao em dùng median thay vì mean?"
> **A:** "Median không bị ảnh hưởng bởi outliers như mean. Khi dữ liệu có nhiều giá trị bất thường, median an toàn hơn."

> **Q:** "Tại sao em dùng median theo từng nhóm bệnh?"
> **A:** "Vì đặc điểm sinh học: người có bệnh tim thường có cholesterol khác người không bệnh. Nếu dùng chung 1 median sẽ không chính xác."

> **Q:** "Em có thể bỏ luôn 172 dòng đó không?"
> **A:** "Không nên ạ. 172 dòng chiếm ~18.7% dữ liệu, bỏ đi sẽ mất nhiều thông tin. Imputation giúp giữ lại dữ liệu."

---

# 🟢 CELL 5: THỐNG KÊ TUỔI, GIỚI TÍNH, CHESTPAINTYPE

## Lý thuyết: **GroupBy & Crosstab**

### 🔹 Phát hiện từ dữ liệu:

- Tuổi trung bình nhóm có bệnh cao hơn nhóm không bệnh
- Nam giới có tỷ lệ bệnh cao hơn nữ
- ChestPainType ASY (asymptomatic) có tỷ lệ bệnh rất cao

### Câu hỏi cô có thể hỏi:

> **Q:** "Em rút ra được insight gì từ bảng thống kê này?"
> **A:** "Tuổi càng cao nguy cơ càng lớn, nam giới có nguy cơ cao hơn nữ, người đau ngực không điển hình (ASY) có nguy cơ rất cao."

---

# 🟢 CELL 6: KIỂM TRA TÍNH TUYẾN TÍNH

## Lý thuyết: **Tuyến tính vs Phi tuyến**

### 🔹 Ý tưởng:

- **Logistic Regression (tuyến tính)**: tìm đường thẳng phân cách
- **Decision Tree (phi tuyến)**: tìm ngưỡng split
- So sánh accuracy của 2 model → biết dữ liệu có tính tuyến tính không

### 🔹 Cross-validation (cv=5):

- Chia dữ liệu thành 5 phần, train 5 lần, mỗi lần lấy 1 phần test
- Kết quả trung bình đáng tin cậy hơn 1 lần chia

### Câu hỏi cô có thể hỏi:

> **Q:** "Tại sao Logistic Regression lại chạy tốt hơn Decision Tree?"
> **A:** "Vì dữ liệu có xu hướng tuyến tính, các features ảnh hưởng cộng dồn đến bệnh tim chứ không phải dạng cây phân cấp."

> **Q:** "Tuyến tính là gì?"
> **A:** "Là mối quan hệ cộng dồn: kết quả = w1*x1 + w2*x2 + ... + bias. Trái ngược với phi tuyến là dạng cây hoặc neural network có thể học patterns phức tạp hơn."

---

# 🟢 CELL 7: PREPROCESSING

## Lý thuyết: **StandardScaler & OneHotEncoder**

### 🔹 StandardScaler (chuẩn hóa Z-score):

**Công thức:** Z = (x - mean) / std
**Kết quả:** mean = 0, std = 1
**Tại sao cần?** Các thuật toán như KNN, SVM, Logistic Regression nhạy cảm với tỷ lệ features. Feature có đơn vị lớn (VD: Cholesterol ~200) sẽ lấn át feature có đơn vị nhỏ (VD: FastingBS ~0.5).

### 🔹 OneHotEncoder (mã hóa one-hot):

**Ý tưởng:** Biến mỗi giá trị của cột phân loại thành 1 cột 0/1 riêng
**Ví dụ:** ChestPainType = [ATA, NAP, ASY, TA] → 4 cột

### 🔹 drop='first':

- Bỏ cột đầu tiên để tránh **đa cộng tuyến** (multicollinearity)
- Nếu có k cột, chỉ cần k-1 cột là đủ (cột cuối suy ra được)

### 🔹 ColumnTransformer:

- Kết hợp nhiều bước preprocessing trong 1 pipeline
- Tránh rò rỉ dữ liệu (data leakage)

### Câu hỏi cô có thể hỏi:

> **Q:** "StandardScaler hoạt động thế nào?"
> **A:** "Z = (x - mean)/std. Biến dữ liệu về mean=0, std=1. Giúp các features có cùng tỷ lệ, không bị feature có giá trị lớn lấn át."

> **Q:** "Tại sao cần OneHotEncoder?"
> **A:** "Vì machine learning chỉ hiểu số, không hiểu chữ. Mã hóa 0/1 giúp model hiểu được các giá trị phân loại."

> **Q:** "Tại sao có drop='first'?"
> **A:** "Để tránh đa cộng tuyến. Nếu có k giá trị, chỉ cần k-1 cột là đủ. Ví dụ: biết không phải ATA, NAP, TA thì chắc chắn là ASY."

> **Q:** "Data leakage là gì?"
> **A:** "Là khi thông tin từ test set bị rò rỉ vào train set. VD: dùng toàn bộ dữ liệu để tính mean trước khi chia train/test → sai. ColumnTransformer tránh được điều này."

---

# 🟢 CELL 8: CORRELATION MATRIX

## Lý thuyết: **Tương quan (Correlation)**

### 🔹 Hệ số tương quan Pearson:

- Giá trị từ -1 đến +1
- **+1**: tương quan thuận hoàn hảo (cùng tăng, cùng giảm)
- **0**: không có tương quan tuyến tính
- **-1**: tương quan nghịch hoàn hảo

### 🔹 Kết quả thực tế:

- ST_Slope, ExerciseAngina, Oldpeak có tương quan mạnh nhất với HeartDisease
- RestingBP, FastingBS, RestingECG gần như không tương quan

### Câu hỏi cô có thể hỏi:

> **Q:** "Correlation = 0 có nghĩa là 2 biến không có quan hệ gì không?"
> **A:** "Không ạ. Correlation chỉ đo quan hệ tuyến tính. 2 biến có thể có quan hệ phi tuyến mạnh (VD: parabol) nhưng correlation = 0."

> **Q:** "Multicollinearity là gì? Tại sao cần tránh?"
> **A:** "Là hiện tượng 2 features có tương quan rất cao với nhau (|r| > 0.8). Khi đó model bị nhiễu, khó xác định feature nào thực sự quan trọng."

---

# 🟢 CELL 9: HISTOGRAM & BOXPLOT

## Lý thuyết: **Trực quan hóa dữ liệu**

### 🔹 Histogram:

- Biểu đồ phân bố tần suất
- So sánh 2 nhóm: có bệnh (xanh) vs không bệnh (cam)
- KDE: đường mật độ ước lượng

### 🔹 Boxplot:

- Tóm tắt bằng 5 số: min, Q1, median, Q3, max
- Dấu chấm: outliers (giá trị ngoại lai)
- IQR = Q3 - Q1 (độ trải giữa)

### Câu hỏi cô có thể hỏi:

> **Q:** "Boxplot cho em biết gì?"
> **A:** "Cho biết median, độ phân tán (IQR), outliers. So sánh 2 nhóm có bệnh/không bệnh."

> **Q:** "Outlier là gì? Có nên bỏ không?"
> **A:** "Outlier là giá trị khác biệt lớn so với phần còn lại. Có thể bỏ nếu có lý do (lỗi nhập liệu), nhưng cẩn thận vì có thể là thông tin quý giá."

---

# 🟢 CELL 10: DECISION TREE - TỰ CODE

## Lý thuyết: **Decision Tree (Cây quyết định)**

### 🔹 Ý tưởng chính:

Chia dữ liệu thành các nhóm nhỏ dần dựa trên các câu hỏi "Có/Không".
Giống như trò chơi "20 câu hỏi" - mỗi câu hỏi giúp thu hẹp phạm vi.

### 🔹 Cấu trúc cây:

```
Node gốc: Oldpeak <= 0.5?
  ├── Yes: Age <= 50?
  │     ├── Yes: KHÔNG BỆNH (leaf)
  │     └── No:  CÓ BỆNH (leaf)
  └── No:  ExerciseAngina = Y?
        ├── Yes: CÓ BỆNH (leaf)
        └── No:  MaxHR <= 140?
              ├── Yes: CÓ BỆNH (leaf)
              └── No:  KHÔNG BỆNH (leaf)
```

### 🔹 Entropy:

**Công thức:** H(S) = -Σ p_i \* log₂(p_i)

**Ý nghĩa:** Đo độ hỗn loạn của dữ liệu.

- Entropy = 0: tinh khiết (100% 1 class)
- Entropy = 1: hỗn loạn nhất (50-50)

**Ví dụ:**

```
10 mẫu: 5 có bệnh, 5 không bệnh
H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0  (hỗn loạn nhất)

10 mẫu: 10 có bệnh, 0 không bệnh
H = -(1*log2(1) + 0*log2(0)) = 0  (tinh khiết)
```

### 🔹 Information Gain:

**Công thức:** IG = H(parent) - (n_left/n)*H(left) - (n_right/n)*H(right)

**Ý nghĩa:** Đo lường mức độ giảm entropy sau khi split.

**Ví dụ cụ thể:**

```
Parent: 10 mẫu (5 có, 5 không) → H=1.0

Split theo Oldpeak <= 0.5:
  Left (7 mẫu): 2 có, 5 không → H_left = 0.86
  Right (3 mẫu): 3 có, 0 không → H_right = 0

IG = 1.0 - (7/10)*0.86 - (3/10)*0 = 1.0 - 0.602 = 0.398

Split theo MaxHR <= 140:
  Left (5 mẫu): 4 có, 1 không → H_left = 0.72
  Right (5 mẫu): 1 có, 4 không → H_right = 0.72

IG = 1.0 - (5/10)*0.72 - (5/10)*0.72 = 1.0 - 0.72 = 0.28

→ Chọn split Oldpeak vì IG lớn hơn!
```

### 🔹 Cách xây dựng cây (đệ quy - recursion):

```
Hàm build_tree(X, y, depth):
  1. Nếu depth >= max_depth → tạo leaf (lấy class đa số)
  2. Nếu số mẫu < min_samples_split → tạo leaf
  3. Nếu tất cả cùng class → tạo leaf
  4. Tìm best split (feature, threshold) có IG lớn nhất
  5. Chia dữ liệu: left = X <= threshold, right = X > threshold
  6. left_child = build_tree(left_data, depth+1)
  7. right_child = build_tree(right_data, depth+1)
  8. Trả về node chứa feature, threshold, left, right
```

### 🔹 Cách dự đoán (traverse):

```
Hàm predict(x, node):
  1. Nếu node là leaf → trả về value
  2. Nếu x[node.feature] <= node.threshold → predict(x, node.left)
  3. Ngược lại → predict(x, node.right)
```

### Câu hỏi cô có thể hỏi:

> **Q:** "Decision Tree hoạt động thế nào?"
> **A:** "Nó xây dựng cây quyết định bằng cách liên tục chọn feature và ngưỡng split có Information Gain lớn nhất. Mỗi node là 1 câu hỏi, leaf node là kết quả cuối cùng."

> **Q:** "Entropy là gì?"
> **A:** "Là độ đo sự hỗn loạn. Entropy = 0 khi dữ liệu thuần nhất (1 class), Entropy = 1 khi 50-50."

> **Q:** "Information Gain là gì?"
> **A:** "Là mức độ giảm entropy sau khi split. IG càng lớn, split càng tốt. Công thức: IG = H(parent) - weighted H(children)."

> **Q:** "max_depth và min_samples_split có tác dụng gì?"
> **A:** "Cả 2 đều để chống overfitting. max_depth giới hạn độ sâu cây. min_samples_split yêu cầu tối thiểu số mẫu để split. Nếu không có, cây có thể học đến từng mẫu → overfit."

> **Q:** "Overfitting là gì?"
> **A:** "Là hiện tượng model học quá khớp với dữ liệu train, nhưng không tổng quát hóa được cho dữ liệu mới. Giống như học vẹt. Decision Tree rất dễ overfit nếu để cây quá sâu."

> **Q:** "Recursion là gì và nó được dùng thế nào trong code?"
> **A:** "Recursion là hàm tự gọi chính nó. Trong Decision Tree, \_build_tree gọi lại chính nó để xây cây con bên trái và bên phải. Điều kiện dừng là khi đạt max_depth, không đủ mẫu, hoặc pure node."

---

# 🟢 CELL 11: NAIVE BAYES - TỰ CODE

## Lý thuyết: **Naive Bayes**

### 🔹 Định lý Bayes (CỐT LÕI):

```
P(y | x) = P(x | y) * P(y) / P(x)

Trong đó:
- P(y | x): posterior - xác suất class y khi biết features x
- P(x | y): likelihood - khả năng features x xuất hiện ở class y
- P(y): prior - xác suất tiên nghiệm của class y
- P(x): evidence - xác suất của features x
```

### 🔹 Naive Bayes cho bài toán:

```
P(Có bệnh | features) ∝ P(Có bệnh) × P(f₁ | Có bệnh) × ... × P(f₁₅ | Có bệnh)

Vì giả định "naive": các features độc lập với nhau
→ P(x₁, x₂, ..., x₁₅ | y) = P(x₁ | y) × P(x₂ | y) × ... × P(x₁₅ | y)
```

### 🔹 Prior (xác suất tiên nghiệm):

```
P(Không bệnh) = 410/918 = 0.447
P(Có bệnh)    = 508/918 = 0.553
```

→ Không có bệnh thì khả năng không bệnh cao hơn một chút

### 🔹 Gaussian PDF (Probability Density Function):

Vì features là số thực (đã StandardScaler), dùng Gaussian:

```
f(x | mean, var) = 1/√(2π×var) × exp(-(x-mean)²/(2×var))
```

**Ví dụ với Oldpeak:**

```
Nhóm không bệnh: mean=0.41, var=0.36
  P(Oldpeak=1.5 | Không bệnh) = GaussianPDF(1.5, 0.41, 0.36) = 0.128

Nhóm có bệnh: mean=1.27, var=1.14
  P(Oldpeak=1.5 | Có bệnh) = GaussianPDF(1.5, 1.27, 1.14) = 0.365

→ Oldpeak=1.5 có khả năng xuất hiện ở người có bệnh cao hơn!
```

### 🔹 Tại sao dùng log?

```
Với 15 features, mỗi P < 1:
P = 0.5 × 0.3 × 0.7 × ... × 0.4 = 0.0000001 (underflow!)

log(P) = log(0.5) + log(0.3) + ... → ổn định, không underflow
```

### Câu hỏi cô có thể hỏi:

> **Q:** "Naive Bayes dựa trên định lý nào?"
> **A:** "Định lý Bayes: P(y|x) = P(x|y)\*P(y)/P(x). Tính xác suất class dựa trên features."

> **Q:** "'Naive' ở đây nghĩa là gì?"
> **A:** "Naive = ngây thơ, vì giả định các features độc lập với nhau. Điều này hiếm khi đúng trong thực tế, nhưng thuật toán vẫn chạy tốt."

> **Q:** "Tại sao dùng Gaussian distribution?"
> **A:** "Vì features của chúng em là số thực (sau StandardScaler), phù hợp với phân phối Gaussian. Công thức: f = 1/sqrt(2π*var) * exp(-(x-mean)²/(2\*var))."

> **Q:** "Tại sao thêm 1e-9 vào variance?"
> **A:** "Để tránh chia cho 0. Nếu 1 feature không thay đổi trong 1 class, var=0 → công thức Gaussian bị lỗi."

---

# 🟢 CELL 12: HÀM EVALUATION METRICS

## Lý thuyết: **Confusion Matrix & Metrics**

### 🔹 Confusion Matrix (Ma trận nhầm lẫn):

```
              Dự đoán
              0     1
Thực tế 0  [ TN    FP ]
         1  [ FN    TP ]
```

| Ký hiệu | Tên            | Ý nghĩa                            | Số (VD) |
| ------- | -------------- | ---------------------------------- | ------- |
| **TN**  | True Negative  | Đúng: không bệnh → đoán không bệnh | 67      |
| **FP**  | False Positive | Sai: không bệnh → đoán có bệnh     | 10      |
| **FN**  | False Negative | Sai: có bệnh → đoán không bệnh     | 22      |
| **TP**  | True Positive  | Đúng: có bệnh → đoán có bệnh       | 84      |

### 🔹 Các metrics quan trọng:

**Accuracy:** (TP + TN) / Total = (84+67)/183 = 0.825

- Tỷ lệ dự đoán đúng trên tổng số
- Dễ bị "ảo" nếu mất cân bằng

**Precision:** TP / (TP + FP) = 84/(84+10) = 0.894

- Khi nói "có bệnh", đúng bao nhiêu %?
- Quan trọng khi FP gây hại (VD: báo động giả)

**Recall (Sensitivity):** TP / (TP + FN) = 84/(84+22) = 0.792

- Phát hiện được bao nhiêu % người bệnh?
- Quan trọng khi FN gây hại (VD: bỏ sót người bệnh)

**F1-Score:** 2 × (P × R) / (P + R) = 2×(0.894×0.792)/(0.894+0.792) = 0.840

- Cân bằng giữa Precision và Recall

### Câu hỏi cô có thể hỏi:

> **Q:** "Tại sao không chỉ dùng Accuracy?"
> **A:** "Accuracy dễ bị ảo nếu mất cân bằng. VD: 90% không bệnh, model đoán ai cũng không bệnh → accuracy 90% nhưng model vô dụng."

> **Q:** "Precision và Recall khác nhau thế nào?"
> **A:** "Precision là trong số những người được đoán có bệnh, bao nhiêu % đúng. Recall là trong số người thực sự có bệnh, model phát hiện được bao nhiêu %."

> **Q:** "Khi nào cần Recall cao hơn Precision?"
> **A:** "Trong y tế, cần Recall cao vì bỏ sót người bệnh (FN) nguy hiểm hơn báo động giả (FP)."

---

# 🟢 CELL 13-14: TRAIN DECISION TREE + TUNING

## Lý thuyết: **Train/Test Split & Hyperparameter Tuning**

### 🔹 Train/Test Split:

- **Train**: 735 mẫu (80%) → để học
- **Test**: 183 mẫu (20%) → để đánh giá
- **random_state=42**: đảm bảo reproducibility

### 🔹 Hyperparameter Tuning (7 tổ hợp tham số):

```
max_depth=3, min_samples=2  → Accuracy: 0.8361
max_depth=5, min_samples=2  → Accuracy: 0.8361
max_depth=5, min_samples=10 → Accuracy: 0.8251
max_depth=10, min_samples=2 → Accuracy: 0.8251
max_depth=10, min_samples=20 → Accuracy: 0.8251 (best F1)
max_depth=15, min_samples=20 → Accuracy: 0.8251
max_depth=20, min_samples=50 → Accuracy: 0.8251
```

### Câu hỏi cô có thể hỏi:

> **Q:** "Tại sao cần chia train/test?"
> **A:** "Để đánh giá model trên dữ liệu chưa thấy. Nếu đánh giá ngay trên train, model có thể overfit nhưng vẫn cho accuracy cao."

> **Q:** "Hyperparameter là gì? Khác gì parameter?"
> **A:** "Hyperparameter do người dùng đặt trước khi train (VD: max_depth). Parameter do model tự học từ dữ liệu (VD: weight, threshold)."

> **Q:** "random_state=42 có tác dụng gì?"
> **A:** "Đảm bảo kết quả có thể tái tạo được. Cùng seed → cùng cách chia → cùng kết quả."

---

# 🟢 CELL 15: CONFUSION MATRIX & K-FOLD CV

## Lý thuyết: **K-Fold Cross Validation**

### 🔹 K-Fold (k=5):

- Chia dữ liệu thành 5 phần bằng nhau
- Train 5 lần, mỗi lần 4 phần train + 1 phần test
- Kết quả: trung bình ± độ lệch chuẩn

### 🔹 Tại sao dùng K-Fold?

- **Đánh giá ổn định hơn** chỉ 1 lần chia train/test
- **Phát hiện overfitting**: nếu kết quả các fold chênh lệch lớn → model không ổn định

### Câu hỏi cô có thể hỏi:

> **Q:** "K-Fold khác gì Train/Test Split?"
> **A:** "Train/Test chỉ đánh giá 1 lần → kết quả có thể may rủi. K-Fold đánh giá k lần → trung bình đáng tin cậy hơn."

> **Q:** "Std trong K-Fold nói lên điều gì?"
> **A:** "Std nhỏ → model ổn định (kết quả ít thay đổi giữa các fold). Std lớn → model không ổn định, phụ thuộc vào cách chia dữ liệu."

---

# 🟢 CELL 16: NAIVE BAYES EVALUATION

## Lý thuyết: **So sánh 2 models**

### 🔹 Kết quả Naive Bayes:

```
Accuracy: 0.8470
Precision: 0.9149 (cao hơn DT)
Recall: 0.8113 (cao hơn DT)
F1-Score: 0.8600 (cao hơn DT)
```

### 🔹 Naive Bayes vs Decision Tree:

| Tiêu chí   | Naive Bayes      | Decision Tree     |
| ---------- | ---------------- | ----------------- |
| Cơ sở      | Xác suất (Bayes) | Quy tắc (Entropy) |
| Giả định   | Features độc lập | Không             |
| Tuning     | Không cần        | Cần tuning        |
| Overfit    | Ít               | Nhiều             |
| Giải thích | Khó (xác suất)   | Dễ (cây)          |

---

# 🟢 CELL 17: SO SÁNH 8 MODELS

## Lý thuyết: **Các thuật toán mở rộng**

### 1️⃣ Logistic Regression 🏆 (F1: 0.878)

**Ý tưởng:** Dùng hàm sigmoid để chuyển tổng có trọng số thành xác suất.

```
z = w₁x₁ + w₂x₂ + ... + bias
P(có bệnh) = 1/(1+e^(-z))
Nếu P >= 0.5 → class 1
```

- ✅ Cho xác suất, dễ giải thích
- ✅ Ít tham số → khó overfit
- ❌ Chỉ học được quan hệ tuyến tính

### 2️⃣ KNN (F1: 0.876)

**Ý tưởng:** "Dựa vào hàng xóm để phán đoán."

- Tính khoảng cách đến K người gần nhất
- Lấy đa số phiếu
- K=5: mượt mà, không quá nhạy cảm với nhiễu
- ✅ Không cần train (lazy learner)
- ❌ Chậm khi dữ liệu lớn

### 3️⃣ Random Forest (F1: 0.876)

**Ý tưởng:** "Nhiều cây yếu → 1 rừng mạnh."

- Xây 100 Decision Trees, mỗi cây trên 1 subset dữ liệu
- Lấy biểu quyết đa số
- ✅ Mạnh, chống overfit tốt
- ❌ Chậm, khó giải thích

### 4️⃣ SVM (F1: 0.850)

**Ý tưởng:** "Tìm đường phân cách có margin rộng nhất."

- Chỉ quan tâm đến các điểm gần ranh giới (support vectors)
- ✅ Mạnh với dữ liệu nhiều chiều
- ❌ Không cho xác suất (mặc định)

### Câu hỏi cô có thể hỏi:

> **Q:** "Tại sao Logistic Regression lại tốt nhất?"
> **A:** "Vì dữ liệu có tính tuyến tính (đã kiểm tra ở Cell 6). Logistic Regression ít tham số, khó overfit, phù hợp với dữ liệu 918 mẫu."

> **Q:** "KNN hoạt động thế nào?"
> **A:** "Tính khoảng cách từ mẫu mới đến tất cả mẫu train, chọn K mẫu gần nhất, lấy đa số phiếu. K=5 là phổ biến."

> **Q:** "Random Forest khác gì Decision Tree?"
> **A:** "Random Forest dùng nhiều cây (ensemble), mỗi cây học trên subset dữ liệu khác nhau. Giảm overfit so với 1 cây đơn."

> **Q:** "SVM là gì?"
> **A:** "Support Vector Machine - tìm siêu phẳng phân cách 2 lớp với margin lớn nhất. Chỉ dùng các điểm support vectors gần ranh giới."

---

# 🎯 TỔNG KẾT - CÁC CÂU HỎI CÓ THẾ HỎI

## Kiến thức nền:

1. Supervised vs Unsupervised learning?
2. Classification vs Regression?
3. Overfitting vs Underfitting?
4. Bias vs Variance tradeoff?
5. Train/Test/Validation split?

## Preprocessing:

6. StandardScaler vs MinMaxScaler?
7. OneHotEncoder vs LabelEncoder?
8. Data leakage là gì?
9. Missing value xử lý thế nào?

## Decision Tree:

10. Entropy là gì? Công thức?
11. Information Gain là gì?
12. Gini Impurity khác Entropy thế nào?
13. max_depth và min_samples_split để làm gì?
14. Tại sao Decision Tree dễ overfit?

## Naive Bayes:

15. Định lý Bayes? Công thức?
16. "Naive" ở đây là gì?
17. Gaussian Naive Bayes hoạt động thế nào?
18. Tại sao phải dùng log?

## Evaluation:

19. Confusion Matrix gồm những gì?
20. Accuracy vs Precision vs Recall vs F1?
21. K-Fold Cross Validation?
22. Khi nào dùng metric nào?

## Models mở rộng:

23. Logistic Regression hoạt động thế nào?
24. KNN là gì? Lazy learner là gì?
25. Random Forest là gì? Ensemble là gì?
26. SVM là gì? Support Vector là gì?
    </｜｜DSML｜｜parameter>
    </write_to_file>
