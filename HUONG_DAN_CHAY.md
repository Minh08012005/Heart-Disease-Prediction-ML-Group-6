# 📋 Hướng Dẫn Chạy Dự Án Dự Đoán Bệnh Tim

## 🎯 Giới Thiệu Dự Án

Dự án này sử dụng **Machine Learning** để dự đoán bệnh tim từ các chỉ số y tế của bệnh nhân.

- **Dataset**: 918 mẫu bệnh nhân với 11 chỉ số y tế
- **Mục tiêu**: Phân loại nhị phân (có/không bệnh tim)
- **Model tốt nhất**: Logistic Regression (F1-Score: 0.8780)

---

## 💬 Câu Hỏi Thường Gặp

**"Nếu train các thuật toán rồi thì đưa dữ liệu dự đoán vào, kết quả trực quan ở đâu?"**

→ **Xem file:** `GIAI_THICH_QUY_TRINH.md` để hiểu rõ quy trình và kết quả!

**Tóm tắt:**

- ✅ Metrics & confusion matrix → Terminal (khi chạy `python main.py`)
- ✅ Biểu đồ chi tiết → Notebooks (01_EDA, 03_Visualization, 04_Train_Evaluate)
- ✅ Model registry → JSON & Markdown files

---

**"Tại sao Decision Tree với tham số khác nhau lại cho kết quả khác nhau?"**

→ **Xem file:** `THAM_SO_VA_K_FOLD.md` để hiểu về tuning tham số và K-Fold!

**Tóm tắt:**

- ✅ max_depth & min_samples_split là gì
- ✅ Tại sao Naive Bayes không cần tuning
- ✅ K-Fold Cross Validation & stability của model
- ✅ Kết quả thực tế từ dự án

---

## ⚠️ Lưu Ý: Models Được Generate Locally

**Models KHÔNG được lưu trên GitHub!**

**Tại sao?**

- File `.pkl` là binary format - không thể xem được
- Giữ repo sạch sẽ & nhỏ gọn
- Đảm bảo reproducibility

**✅ Thay vào đó:**

- 📄 `MODEL_REGISTRY.json` - Metadata & metrics (xem được trên GitHub)
- 📚 `MODEL_REGISTRY.md` - Hướng dẫn chi tiết
- 🔄 Models được **generate** khi chạy `python main.py`

**→ Xem chi tiết tại `MODEL_REGISTRY.md`**

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
- ✅ **Generate & lưu models vào `models_output/` (tạo lúc chạy)**
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

**💾 Models được lưu vào:**

```
models_output/
├── custom_dt_20260526_093507.pkl
├── custom_nb_20260526_093507.pkl
├── sklearn_dt_20260526_093507.pkl
└── [các models khác...]
```

### **🔷 Cách 2: Huấn Luyện Và Lưu Model**

```bash
python scripts/train.py --model decision_tree
```

**Tuỳ chọn models:**

- `decision_tree` - Custom Decision Tree
- `naive_bayes` - Custom Naive Bayes

**Kết quả:**

- Model sẽ được **generate & lưu** vào `models_output/` với tên timestamp
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

**💡 Mẹo:** Liệt kê các models đã tạo:

```bash
ls -lh models_output/
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

## 📊 Xem Thông Tin Models (Model Registry)

Sau khi chạy `python main.py`, bạn có thể xem metadata của tất cả models:

### **Cách 1: Xem File JSON (trên GitHub hoặc cục bộ)**

```bash
cat MODEL_REGISTRY.json
```

Hoặc mở trong editor: `MODEL_REGISTRY.json`

Sẽ hiển thị:

```json
{
  "models": [
    {
      "id": "logistic_regression_best",
      "name": "Logistic Regression (Sklearn)",
      "metrics": {
        "accuracy": 0.8634,
        "f1_score": 0.8780
      },
      "rank": "🏆 Best Model"
    },
    ...
  ]
}
```

### **Cách 2: Đọc Hướng Dẫn Chi Tiết**

```bash
# Xem file hướng dẫn
cat MODEL_REGISTRY.md

# Hoặc trên GitHub:
# https://github.com/.../MODEL_REGISTRY.md
```

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
├── MODEL_REGISTRY.json               # Metadata của models (xem được)
├── MODEL_REGISTRY.md                 # Hướng dẫn chi tiết
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
├── models_output/                   # ⚠️ Folder lưu models (KHÔNG tracked GitHub)
│   ├── custom_dt_20260526_093507.pkl      ← Generated by python main.py
│   ├── custom_nb_20260526_093507.pkl      ← Generated locally
│   └── [các models mới được thêm khi chạy scripts]
│
│   💡 Lưu ý: Những file .pkl này KHÔNG được lưu trên GitHub
│      Chúng được generate tự động khi chạy `python main.py`
│      Xem MODEL_REGISTRY.json & MODEL_REGISTRY.md để biết chi tiết
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
python main.py  # Kiểm tra xem tất cả đã setup đúng chưa + Generate models
```

### **Bước 2️⃣: Xem thông tin models đã tạo**

```bash
# Xem metadata (models không được track GitHub, chỉ metadata được track)
cat MODEL_REGISTRY.json

# Hoặc xem hướng dẫn chi tiết
cat MODEL_REGISTRY.md

# Xem danh sách models cục bộ
ls -lh models_output/
```

### **Bước 3️⃣: Xem phân tích dữ liệu (Optional)**

```bash
jupyter notebook notebooks/01_EDA.ipynb
# Xem thống kê, biểu đồ, phân tích dữ liệu
```

### **Bước 4️⃣: Huấn luyện model mới (Optional)**

```bash
python scripts/train.py --model naive_bayes
# Model được generate & lưu vào models_output/ với timestamp
```

### **Bước 5️⃣: Đánh giá model**

```bash
python scripts/predict.py --model "models_output/naive_bayes_20260526_093507.pkl"
```

### **Bước 6️⃣: Xem so sánh tất cả models**

```bash
python main.py
# Xem bảng so sánh tất cả 8 models + Generate tất cả models
```

### **Bước 7️⃣: Xem thông tin & metadata models**

```bash
# Xem các models được generate
ls -lh models_output/

# Xem metrics & metadata (JSON format - xem được)
cat MODEL_REGISTRY.json

# Hoặc xem hướng dẫn chi tiết
cat MODEL_REGISTRY.md
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

# Chạy main.py để generate lại models
python main.py

# Dùng đường dẫn đúng khi chạy predict.py
python scripts/predict.py --model "models_output/logistic_regression_*.pkl"
```

### ❌ Lỗi: "Models không được lưu trên GitHub?"

**Giải thích:**

Models (`.pkl` files) là **binary format** - không thể xem trên GitHub. Thay vào đó:

- ✅ **Metadata được lưu**: `MODEL_REGISTRY.json` (JSON format - xem được)
- ✅ **Hướng dẫn được lưu**: `MODEL_REGISTRY.md` (Markdown - xem được)
- ✅ **Models được generate**: Tự động khi chạy `python main.py`

Xem `MODEL_REGISTRY.md` để biết chi tiết!

---

## 📈 Kết Luận

- **Fastest way** 🚀: `python main.py` (2 phút) → Generate models + Compare
- **Detailed way** 📚: Notebooks + `main.py` (30 phút) → Phân tích + Results
- **Custom way** 🔧: `train.py` → `predict.py` (tùy chỉnh) → Train specific models

**💡 Important Notes:**

- Models KHÔNG tracked GitHub (binary files)
- Metadata được tracked (JSON format)
- Generate models locally bằng `python main.py`
- Xem `MODEL_REGISTRY.md` để biết chi tiết cách sử dụng

---

## 👥 Liên Hệ & Support

Nếu có vấn đề, kiểm tra:

1. ✅ Python version >= 3.8
2. ✅ Tất cả dependencies trong requirements.txt
3. ✅ Đường dẫn file đúng
4. ✅ Permissions để ghi file vào `models_output/`

---

**Happy Predicting! 🎉**
