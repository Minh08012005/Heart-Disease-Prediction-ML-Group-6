# 📋 Hướng Dẫn Chạy Dự Án Dự Đoán Bệnh Tim

## 🎯 Giới Thiệu Dự Án

Dự án này sử dụng **Machine Learning** để dự đoán bệnh tim từ các chỉ số y tế của bệnh nhân.

- **Dataset**: 918 mẫu bệnh nhân với 11 chỉ số y tế
- **Mục tiêu**: Phân loại nhị phân (có/không bệnh tim)
- **Model tốt nhất**: Logistic Regression (F1-Score: 0.8780)

---

## 📦 Cài Đặt Môi Trường

### 1️⃣ Clone dự án và vào thư mục

```bash
cd Heart-Disease-Prediction-ML-Group-6
```

### 2️⃣ Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**File requirements.txt chứa:**

- pandas, numpy (xử lý dữ liệu)
- scikit-learn (các model ML)
- matplotlib, seaborn (visualize)
- jupyter (chạy notebooks)

---

## 🚀 Các Cách Chạy Dự Án

### **🔷 Cách 1: Chạy Pipeline Hoàn Chỉnh (Recommended)**

```bash
python main.py
```

**Kết quả:**

- ✅ Load dữ liệu (918 mẫu)
- ✅ Huấn luyện 8 model (2 custom + 6 sklearn)
- ✅ So sánh hiệu suất tất cả models
- ✅ Xác định model tốt nhất
- ✅ Hiển thị confusion matrix

**Output:**

```
======================================================================
  HEART DISEASE PREDICTION PROJECT
  Complete ML Pipeline Execution
======================================================================

STEP 1: Load Data
[OK] Loaded 918 samples
[OK] Train/Test split: 735 train, 183 test

STEP 2: Train Models
[TRAIN] Training Custom Decision Tree...
[TRAIN] Training Custom Naive Bayes...
[TRAIN] Training Sklearn models...

STEP 3: Model Comparison
Model                          Accuracy     Precision    Recall
...

STEP 4: Confusion Matrix Analysis
Custom Decision Tree:
   TN: 66, FP: 11, FN: 26, TP: 80

STEP 5: Summary
   [OK] Data loaded: 918 samples
   [OK] Models trained: 8 models
   [OK] Best model: Logistic Regression (F1: 0.8780)
```

---

### **🔷 Cách 2: Huấn Luyện Và Lưu Model**

```bash
python scripts/train.py --model decision_tree
```

**Tuỳ chọn models:**

- `decision_tree` - Custom Decision Tree
- `naive_bayes` - Custom Naive Bayes

**Kết quả:**

- Model sẽ được lưu vào `models_output/` với tên timestamp
- Ví dụ: `decision_tree_20260526_093507.pkl`

**Ví dụ đầy đủ:**

```bash
python scripts/train.py --model decision_tree --output models_output
```

---

### **🔷 Cách 3: Đánh Giá Model Đã Lưu**

```bash
python scripts/predict.py --model "models_output/decision_tree_20260526_093507.pkl"
```

**Kết quả:**

```
============================================================
📊 EVALUATION RESULTS
============================================================

 Metrics:
   Accuracy:  0.7978 (79.78%)
   Precision: 0.8791
   Recall:    0.7547
   F1-Score:  0.8122

 Confusion Matrix:
   TN: 66, FP: 11, FN: 26, TP: 80

 Interpretation:
   False Negative Rate: 24.53% (missed 26 diseased)
   False Positive Rate: 14.29% (false alarms: 11)
```

---

### **🔷 Cách 4: Phân Tích Chi Tiết Với Notebooks**

Nếu muốn xem quá trình EDA (Exploratory Data Analysis), preprocessing, training chi tiết:

```bash
jupyter notebook
```

Mở các file notebook theo thứ tự:

| #   | Notebook                            | Nội Dung                          |
| --- | ----------------------------------- | --------------------------------- |
| 1️⃣  | `notebooks/01_EDA.ipynb`            | Khám phá dữ liệu, thống kê cơ bản |
| 2️⃣  | `notebooks/02_Preprocessing.ipynb`  | Làm sạch, xử lý dữ liệu           |
| 3️⃣  | `notebooks/03_Visualization.ipynb`  | Vẽ biểu đồ, phân tích hình ảnh    |
| 4️⃣  | `notebooks/04_Train_Evaluate.ipynb` | Huấn luyện & đánh giá models      |
| 5️⃣  | `notebooks/05_Compare_Models.ipynb` | So sánh tất cả models             |

---

## 📊 Kết Quả Mong Đợi

### Hiệu Suất Các Model (trên Test Set)

| Model                   | Accuracy   | Precision  | Recall     | F1-Score      |
| ----------------------- | ---------- | ---------- | ---------- | ------------- |
| **Logistic Regression** | **0.8634** | **0.9091** | **0.8491** | **0.8780** ⭐ |
| KNN (k=5)               | 0.8579     | 0.8846     | 0.8679     | 0.8762        |
| Random Forest           | 0.8579     | 0.8846     | 0.8679     | 0.8762        |
| Custom Naive Bayes      | 0.8470     | 0.9149     | 0.8113     | 0.8600        |
| SVM                     | 0.8306     | 0.8713     | 0.8302     | 0.8502        |
| Sklearn Decision Tree   | 0.8033     | 0.8977     | 0.7453     | 0.8144        |
| Custom Decision Tree    | 0.7978     | 0.8791     | 0.7547     | 0.8122        |
| Sklearn Naive Bayes     | 0.8470     | 0.9149     | 0.8113     | 0.8600        |

---

## 📁 Cấu Trúc Thư Mục

```
├── main.py                           # Pipeline chính
├── config.py                         # Cấu hình tổng
├── requirements.txt                  # Dependencies
│
├── data/
│   ├── heart.csv                    # Dữ liệu gốc
│   ├── heart_cleaned.csv            # Dữ liệu đã làm sạch
│   └── heart_preprocessed.csv       # Dữ liệu đã xử lý (input cho model)
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Phân tích dữ liệu
│   ├── 02_Preprocessing.ipynb       # Xử lý dữ liệu
│   ├── 03_Visualization.ipynb       # Vẽ biểu đồ
│   ├── 04_Train_Evaluate.ipynb      # Huấn luyện & đánh giá
│   └── 05_Compare_Models.ipynb      # So sánh models
│
├── scripts/
│   ├── train.py                     # Script huấn luyện model
│   └── predict.py                   # Script đánh giá model
│
├── src/
│   ├── models/
│   │   ├── decision_tree.py         # Decision Tree từ scratch
│   │   └── naive_bayes.py           # Naive Bayes từ scratch
│   ├── data/
│   │   ├── loader.py                # Hàm load dữ liệu
│   │   └── preprocessor.py          # Xử lý dữ liệu
│   ├── utils.py                     # Metrics & hàm tiện ích
│   └── preprocessing.py
│
├── models_output/                   # Folder lưu models đã train
│   └── decision_tree_20260526_093507.pkl
│
└── reports/                         # Báo cáo chi tiết
    ├── PHAN_1_GIOI_THIEU.md
    ├── PHAN_2_CO_SO_LY_THUYET.md
    ├── PHAN_3_EDA_PREPROCESSING.md
    ├── PHAN_4_TRUC_QUAN_HOA.md
    ├── PHAN_5_THUAT_TOAN.md
    ├── PHAN_6_SO_SANH_DANH_GIA.md
    └── PHAN_7_KET_LUAN.md
```

---

## 💡 Hướng Dẫn Từng Bước Chi Tiết

### **Bước 1️⃣: Cài đặt & kiểm tra**

```bash
pip install -r requirements.txt
python main.py  # Kiểm tra xem tất cả đã setup đúng chưa
```

### **Bước 2️⃣: Xem phân tích dữ liệu (Optional)**

```bash
jupyter notebook notebooks/01_EDA.ipynb
# Xem thống kê, biểu đồ, phân tích dữ liệu
```

### **Bước 3️⃣: Huấn luyện model mới**

```bash
python scripts/train.py --model naive_bayes
# Lưu ý: model được lưu vào models_output/ với timestamp
```

### **Bước 4️⃣: Đánh giá model**

```bash
python scripts/predict.py --model "models_output/naive_bayes_20260526_093507.pkl"
```

### **Bước 5️⃣: Xem so sánh tất cả models**

```bash
python main.py
# Xem bảng so sánh tất cả 8 models
```

---

## ⚙️ Cấu Hình (Config.py)

Có thể tuỳ chỉnh trong file `config.py`:

```python
# Hyperparameter Decision Tree
MODELS_CONFIG = {
    "decision_tree": {
        "max_depth": 10,              # Độ sâu cây
        "min_samples_split": 20       # Số mẫu tối thiểu để split
    },
    "naive_bayes": {},                # Không có hyperparameter
    ...
}

# Tham số tách train/test
TEST_SIZE = 0.2                       # 20% test, 80% train
RANDOM_STATE = 42                     # Seed cho reproducibility
K_FOLD_SPLITS = 5                     # K-Fold Cross Validation
```

---

## 🔧 Troubleshooting

### ❌ Lỗi: `ModuleNotFoundError: No module named 'sklearn'`

**Giải pháp:**

```bash
pip install scikit-learn
```

### ❌ Lỗi: `FileNotFoundError: data/heart_preprocessed.csv`

**Giải pháp:**

- Kiểm tra file có tồn tại trong thư mục `data/`
- Chạy `notebooks/02_Preprocessing.ipynb` để tạo file này

### ❌ Lỗi: Model file không tìm thấy

**Giải pháp:**

```bash
# Kiểm tra file trong models_output/
ls models_output/
# Dùng đường dẫn đúng khi chạy predict.py
```

---

## 📈 Kết Luận

- **Fastest way** 🚀: `python main.py` (2 phút)
- **Detailed way** 📚: Notebooks + `main.py` (30 phút)
- **Custom way** 🔧: `train.py` → `predict.py` (tùy chỉnh)

---

## 👥 Liên Hệ & Support

Nếu có vấn đề, kiểm tra:

1. ✅ Python version >= 3.8
2. ✅ Tất cả dependencies trong requirements.txt
3. ✅ Đường dẫn file đúng
4. ✅ Permissions để ghi file vào `models_output/`

---

**Happy Predicting! 🎉**
