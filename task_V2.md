'''

# ⚡ SPRINT 2: DECISION TREE TRAINING & EVALUATION (30/4 - 4/5)

---

## 🎯 TỔNG QUAN HƯỚNG ĐI

Chúng ta đã hoàn thành:

- ✅ **EDA & Preprocessing** (Minh) - Phase 1
- ✅ **Decision Tree Implementation** (Hiếu & Phong) - Đã merge vào `main`

Bây giờ, **cả 4 thành viên** sẽ cùng nhau tham gia vào quá trình **Training & Tuning Decision Tree**.

### Mục tiêu chung:

```
Mỗi người tự tạo 1 notebook riêng → tự train, tự tuning, tự so sánh với sklearn
→ Leader tổng hợp kết quả tốt nhất từ cả 4 người
```

### Lộ trình tổng thể:

```
Phase 1: EDA & Preprocessing (Minh)          ████████████████ 100% ✅
Phase 2: Decision Tree code (Hiếu & Phong)   ████████████████ 100% ✅
Phase 3: DT Training & Tuning (Cả team)      ░░░░░░░░░░░░░░░░ 0% ⏳ (BẮT ĐẦU)
Phase 4: Naive Bayes (Tuân)                  ░░░░░░░░░░░░░░░░ 0% ❓
Phase 5: So sánh & Báo cáo (All)             ░░░░░░░░░░░░░░░░ 0%
```

---

## 📋 TASK-04: DECISION TREE TRAINING & EVALUATION

**Assignees:** @Minh, @Hiếu, @Phong, @Tuân
**Deadline:** 4/5/2026 (5 ngày)
**Priority:** HIGH 🔴
**Labels:** training, tuning, decision-tree, notebook
**Status:** To Do

---

### 👥 PHÂN CÔNG

| Thành viên | Nhánh               | Notebook cần tạo              |
| ---------- | ------------------- | ----------------------------- |
| **@Minh**  | `minh/dt-training`  | `notebooks/04_DT_Minh.ipynb`  |
| **@Hiếu**  | `hieu/dt-training`  | `notebooks/04_DT_Hieu.ipynb`  |
| **@Phong** | `phong/dt-training` | `notebooks/04_DT_Phong.ipynb` |
| **@Tuân**  | `tuan/dt-training`  | `notebooks/04_DT_Tuan.ipynb`  |

📌 **Mỗi người 1 nhánh riêng → không sợ conflict!**

---

### 📊 SƠ ĐỒ CẤU TRÚC NOTEBOOK

```
04_DT_TenCaNhan.ipynb
│
├── GIAI ĐOẠN 1: CHUẨN BỊ (Cell 1 → 3)
│   ├── Cell 1: Tiêu đề + Giới thiệu
│   ├── Cell 2: Import thư viện
│   └── Cell 3: Load dữ liệu
│
├── GIAI ĐOẠN 2: TRAIN CƠ BẢN (Cell 4 → 5)
│   ├── Cell 4: Train Custom DT (max_depth=5)
│   └── Cell 5: Đánh giá kết quả
│
├── GIAI ĐOẠN 3: GIẢI THÍCH (Cell 6)
│   └── Cell 6: Giải thích ý nghĩa các metrics
│
├── GIAI ĐOẠN 4: HYPERPARAMETER TUNING (Cell 7 → 8)
│   ├── Cell 7: Tuning tham số
│   └── Cell 8: Vẽ biểu đồ tuning
│
└── GIAI ĐOẠN 5: SO SÁNH VỚI SKLEARN (Cell 9 → 10)
    ├── Cell 9: So sánh với sklearn
    └── Cell 10: Nhận xét & Kết luận
```

---

### 📝 CHI TIẾT TỪNG GIAI ĐOẠN

---

#### GIAI ĐOẠN 1: CHUẨN BỊ (Cell 1 → 3)

**Mục tiêu:** Thiết lập môi trường làm việc, load dữ liệu.

**Cell 1 - Tiêu đề:**

```markdown
# 🌳 DECISION TREE TRAINING - [TÊN CỦA BẠN]

**Mô tả:** Notebook này thực hiện:

1. Huấn luyện Decision Tree tự code trên dữ liệu bệnh tim
2. Đánh giá mô hình bằng các metrics
3. Tuning tham số để tìm kết quả tốt nhất
4. So sánh với thư viện sklearn

**Tác giả:** [Tên của bạn]
**Ngày:** 30/4/2026
```

**Cell 2 - Import thư viện:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom Decision Tree của nhóm
from src.models.decision_tree import DecisionTree

# Các hàm tiện ích tự viết
from src.utils import train_test_split, accuracy_score
from src.utils import confusion_matrix, precision_score
from src.utils import recall_score, f1_score

# Thư viện sklearn để so sánh
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as sk_accuracy
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score as sk_recall
from sklearn.metrics import f1_score as sk_f1
```

**Cell 3 - Load dữ liệu:**

```python
# Đọc dữ liệu đã được preprocessing bởi Minh
df = pd.read_csv('data/heart_preprocessed.csv')

# Tách features (X) và target (y)
X = df.drop('HeartDisease', axis=1).values  # 15 features
y = df['HeartDisease'].values                # 0: không bệnh, 1: có bệnh

print(f"📊 Dữ liệu: {len(X)} mẫu, {X.shape[1]} features")
print(f"   Target: 0(Không bệnh)={sum(y==0)}, 1(Có bệnh)={sum(y==1)}")

# Chia train/test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"✂️  Chia dữ liệu: Train={len(X_train)}, Test={len(X_test)}")
```

---

#### GIAI ĐOẠN 2: TRAIN CƠ BẢN (Cell 4 → 5)

**Mục tiêu:** Train Decision Tree với tham số mặc định, xem kết quả đầu tiên.

**Cell 4 - Train Custom Decision Tree:**

```python
# Tạo mô hình với tham số mặc định
dt = DecisionTree(max_depth=5, min_samples_split=2)

# Huấn luyện trên tập train
dt.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = dt.predict(X_test)

print("✅ Đã huấn luyện xong Custom Decision Tree!")
```

**Cell 5 - Đánh giá kết quả:**

```python
print("=" * 55)
print("📊 KẾT QUẢ CUSTOM DECISION TREE")
print("=" * 55)
print(f"  Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:        {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score:      {f1_score(y_test, y_pred):.4f}")
print("-" * 55)
print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Dự đoán: Không Bệnh   Có Bệnh")
print(f"  Thực tế: Không Bệnh    {cm[0,0]:<8}    {cm[0,1]:<8}")
print(f"           Có Bệnh       {cm[1,0]:<8}    {cm[1,1]:<8}")
print("=" * 55)
```

---

#### GIAI ĐOẠN 3: GIẢI THÍCH (Cell 6)

**Mục tiêu:** Hiểu ý nghĩa các con số vừa tính được.

**Cell 6 - Giải thích (Markdown):**

```markdown
### 📝 Giải thích kết quả

#### Accuracy = ?.????

> "Mô hình dự đoán đúng ?.% tổng số bệnh nhân"
>
> Ví dụ: Có 100 người, mô hình đoán đúng ? người

#### Precision = ?.????

> "Khi mô hình nói 'CÓ BỆNH', có ?.% khả năng là đúng"
>
> Precision cao → ít chẩn đoán nhầm người khỏe thành có bệnh

#### Recall = ?.????

> "Mô hình phát hiện được ?.% số người thực sự có bệnh"
>
> ⚠️ **QUAN TRỌNG TRONG Y TẾ**: Recall cao → ít bỏ sót người bệnh

#### F1-Score = ?.????

> "Cân bằng giữa Precision và Recall"
>
> F1 càng gần 1 → mô hình càng tốt

#### Confusion Matrix

|                   |  Dự đoán: Khỏe  | Dự đoán: Bệnh |
| ----------------- | :-------------: | :-----------: |
| **Thực tế: Khỏe** |  TN = ? (đúng)  | FP = ? (nhầm) |
| **Thực tế: Bệnh** | FN = ? (bỏ sót) | TP = ? (đúng) |
```

---

#### GIAI ĐOẠN 4: HYPERPARAMETER TUNING (Cell 7 → 8)

**Mục tiêu:** Thử nhiều bộ tham số để tìm accuracy cao nhất.

**Cell 7 - Hyperparameter Tuning:**

```python
# Các tham số cần thử
max_depth_values = [3, 5, 7, 10, 15, None]
# Ý nghĩa:
#   3   → cây chỉ được hỏi tối đa 3 câu
#   5   → cây hỏi tối đa 5 câu (mặc định)
#   7   → cây hỏi tối đa 7 câu
#   10  → cây hỏi tối đa 10 câu
#   15  → cây hỏi tối đa 15 câu
#   None → không giới hạn số câu hỏi

min_samples_values = [2, 5, 10, 20]
# Ý nghĩa:
#   2   → chỉ cần 2 người là được hỏi tiếp (mặc định)
#   5   → cần 5 người mới được hỏi tiếp
#   10  → cần 10 người mới được hỏi tiếp
#   20  → cần 20 người mới được hỏi tiếp

print("🔧 HYPERPARAMETER TUNING")
print("=" * 55)
print(f"{'max_depth':<12} {'min_samples':<15} {'Accuracy':<10}")
print("-" * 55)

best_acc = 0
best_params = {}
results = []

for max_depth in max_depth_values:
    for min_samples in min_samples_values:
        # Train với bộ tham số hiện tại
        dt = DecisionTree(max_depth=max_depth,
                         min_samples_split=min_samples)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Lưu kết quả
        results.append({
            'max_depth': max_depth,
            'min_samples': min_samples,
            'accuracy': acc
        })

        # Đánh dấu best
        note = "👑" if acc > best_acc else ""
        print(f"{str(max_depth):<12} {str(min_samples):<15} {acc:.4f}  {note}")

        if acc > best_acc:
            best_acc = acc
            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples
            }

print("-" * 55)
print(f"\n✅ Tham số tốt nhất: {best_params}")
print(f"✅ Accuracy cao nhất: {best_acc:.4f}")
```

**Cell 8 - Vẽ biểu đồ tuning:**

```python
# Chuyển kết quả thành DataFrame để vẽ
df_results = pd.DataFrame(results)

# Tạo 2 biểu đồ cạnh nhau
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ 1: Accuracy thay đổi theo max_depth
for ms in min_samples_values:
    subset = df_results[df_results['min_samples'] == ms]
    depths = [str(d) if d is not None else 'None' for d in subset['max_depth']]
    axes[0].plot(depths, subset['accuracy'], marker='o', label=f'min_samples={ms}')
axes[0].set_xlabel('max_depth')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Ảnh hưởng của max_depth đến Accuracy')
axes[0].legend()
axes[0].grid(True)

# Biểu đồ 2: Accuracy thay đổi theo min_samples_split
for md in max_depth_values:
    subset = df_results[df_results['max_depth'] == md]
    axes[1].plot(subset['min_samples'], subset['accuracy'],
                 marker='s', label=f'max_depth={md}')
axes[1].set_xlabel('min_samples_split')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Ảnh hưởng của min_samples_split đến Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('img/tuning_DT_TenBan.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

#### GIAI ĐOẠN 5: SO SÁNH VỚI SKLEARN (Cell 9 → 10)

**Mục tiêu:** So sánh code tự viết với thư viện chuyên nghiệp sklearn.

**Cell 9 - So sánh với sklearn:**

```python
# Lấy tham số tốt nhất từ tuning
best_md = best_params['max_depth']
best_ms = best_params['min_samples_split']

# === CUSTOM DT (đã tuning) ===
print("🌳 Custom Decision Tree (đã tuning)...")
dt_custom = DecisionTree(max_depth=best_md,
                         min_samples_split=best_ms)
dt_custom.fit(X_train, y_train)
y_pred_custom = dt_custom.predict(X_test)

# === SKLEARN DT (cùng tham số) ===
print("⚙️ Sklearn Decision Tree (cùng tham số)...")
dt_sklearn = DecisionTreeClassifier(
    max_depth=best_md,
    min_samples_split=best_ms,
    random_state=42
)
dt_sklearn.fit(X_train, y_train)
y_pred_sklearn = dt_sklearn.predict(X_test)

# === SO SÁNH ===
print("\n" + "=" * 55)
print("📊 SO SÁNH: CUSTOM DT vs SKLEARN DT")
print("=" * 55)
print(f"{'Metric':<15} {'Custom DT':<18} {'Sklearn DT':<18} {'Chênh lệch'}")
print("-" * 55)

metrics = [
    ('Accuracy',
     accuracy_score(y_test, y_pred_custom),
     sk_accuracy(y_test, y_pred_sklearn)),
    ('Precision',
     precision_score(y_test, y_pred_custom),
     sk_precision(y_test, y_pred_sklearn)),
    ('Recall',
     recall_score(y_test, y_pred_custom),
     sk_recall(y_test, y_pred_sklearn)),
    ('F1-Score',
     f1_score(y_test, y_pred_custom),
     sk_f1(y_test, y_pred_sklearn)),
]

for name, custom, sk in metrics:
    diff = custom - sk
    print(f"{name:<15} {custom:<18.4f} {sk:<18.4f} {diff:+.4f}")

print("=" * 55)
```

**Cell 10 - Nhận xét & Kết luận (Markdown):**

```markdown
### 📝 Nhận xét của [TÊN BẠN]

#### 1. Kết quả Custom Decision Tree

- Accuracy mặc định (max_depth=5): ?.????
- Accuracy sau tuning (max_depth=?, min_samples=?): ?.????
- Cải thiện: ?.???? so với mặc định

#### 2. So sánh với sklearn

- Custom DT: Accuracy = ?.????
- Sklearn DT: Accuracy = ?.????
- Chênh lệch: ?.???? → Custom DT hoạt động [tốt / cần cải thiện]

#### 3. Kết luận

- ✅ Decision Tree tự code hoạt động hiệu quả trên bài toán dự đoán bệnh tim
- ✅ Accuracy đạt ~?% sau khi tuning
- ✅ Gần tương đương với thư viện sklearn (chênh ~?%)
- 💡 Hướng phát triển: Có thể cải thiện bằng cách...
```

---

### 📋 CÁC BƯỚC THỰC HIỆN TRÊN GIT

#### Bước 1: Pull code mới nhất từ main

```bash
git checkout main
git pull origin main
```

#### Bước 2: Tạo nhánh riêng của bạn

```bash
# Ví dụ với Hiếu:
git checkout -b hieu/dt-training

# Ví dụ với Phong:
git checkout -b phong/dt-training

# Ví dụ với Tuân:
git checkout -b tuan/dt-training

# Ví dụ với Minh:
git checkout -b minh/dt-training
```

#### Bước 3: Tạo notebook theo cấu trúc trên

Tạo file `notebooks/04_DT_TenBan.ipynb` với 10 cell như đã mô tả.

#### Bước 4: Commit và Push

```bash
git add notebooks/04_DT_TenBan.ipynb
git commit -m "Hoàn thành DT training - Tên của bạn"
git push origin ten-ban/dt-training
```

#### Bước 5: Tạo Pull Request

Lên GitHub, tạo Pull Request từ nhánh của bạn vào `main`
Gán @Minh làm reviewer.

---

### ✅ SAU KHI CẢ 4 XONG

@Minh sẽ tạo notebook tổng hợp:

```markdown
# 📊 SO SÁNH KẾT QUẢ DECISION TREE

| Thành viên   | Best Accuracy  | Best Params (max_depth, min_samples)   |
| ------------ | -------------- | -------------------------------------- |
| Minh         | ?.????         | (?, ?)                                 |
| Hiếu         | ?.????         | (?, ?)                                 |
| Phong        | ?.????         | (?, ?)                                 |
| Tuân         | ?.????         | (?, ?)                                 |
| ------------ | -------------- | -------------------------------------- |
| **Best**     | **?.????**     | **(?, ?)**                             |
```

---

### 📊 TIẾN ĐỘ DỰ ÁN

```
Phase 1: EDA & Preprocessing (Minh)          ████████████████ 100% ✅
Phase 2: Decision Tree code (Hiếu & Phong)   ████████████████ 100% ✅
Phase 3: DT Training & Tuning (Cả team)      ░░░░░░░░░░░░░░░░ 0% ⏳
Phase 4: Naive Bayes (Tuân)                  ░░░░░░░░░░░░░░░░ 0% ❓
Phase 5: So sánh & Báo cáo (All)             ░░░░░░░░░░░░░░░░ 0%
```

---

### 🚨 LƯU Ý QUAN TRỌNG

1. **Mỗi người 1 nhánh riêng** → không sợ conflict
2. **Tên nhánh có quy tắc**: `tên/nội-dung` (vd: `hieu/dt-training`)
3. **Sau khi xong → tạo Pull Request** vào main, gán @Minh review
4. **Minh sẽ tổng hợp** kết quả tốt nhất từ tất cả
5. **Deadline cứng: 4/5/2026** - không delay để kịp tiến độ
