# 📋 Model Registry - Hướng Dẫn Về Các Models

## ⚠️ Quan Trọng: Models Không Được Lưu Trên GitHub

**Vì sao?**

- File `.pkl` (pickled models) là **binary format** - không thể xem trên GitHub
- Dung lượng lớn không cần thiết cho version control
- Tốt hơn là generate locally để đảm bảo reproducibility

**Thay vào đó:**

- 📄 Tệp này (`MODEL_REGISTRY.md`) chứa **metadata & metrics**
- `MODEL_REGISTRY.json` chứa thông tin đầy đủ (có thể đọc được)
- Scripts tự động generate models khi cần

---

## 🚀 Cách Generate Models Locally

### **Lựa Chọn 1: Train Tất Cả Models (Recommended)**

```bash
python main.py
```

**Kết quả:**

- ✅ 8 models được huấn luyện
- ✅ So sánh tất cả metrics
- ✅ Models được lưu vào `models_output/`
- ⏱️ Thời gian: ~2 phút

**Output:**

```
======================================================================
  HEART DISEASE PREDICTION PROJECT
  Complete ML Pipeline Execution
======================================================================

STEP 3: Model Comparison
Model                          Accuracy     Precision    Recall       F1
...
[BEST] Best model: Logistic Regression
   - F1-Score:  0.8780
```

---

### **Lựa Chọn 2: Train Model Cụ Thể**

```bash
# Train Decision Tree
python scripts/train.py --model decision_tree

# Train Naive Bayes
python scripts/train.py --model naive_bayes
```

**Kết quả:**

- Model được lưu vào `models_output/` với timestamp
- Ví dụ: `decision_tree_20260526_093507.pkl`

---

### **Lựa Chọn 3: Đánh Giá Model Đã Lưu**

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
   TN: 66, FP: 11
   FN: 26, TP: 80
```

---

## 📊 Bảng Thông Tin Models

### **Custom Models (Cài Đặt Từ Scratch)**

| Model                    | Accuracy | Precision | Recall | F1-Score | Hyperparameters                      |
| ------------------------ | -------- | --------- | ------ | -------- | ------------------------------------ |
| **Custom Decision Tree** | 0.7978   | 0.8791    | 0.7547 | 0.8122   | `max_depth=10, min_samples_split=20` |
| **Custom Naive Bayes**   | 0.8470   | 0.9149    | 0.8113 | 0.8600   | None                                 |

### **Sklearn Models (Thư Viện)**

| Model                      | Accuracy | Precision | Recall | F1-Score | Rank    |
| -------------------------- | -------- | --------- | ------ | -------- | ------- |
| **Logistic Regression** ⭐ | 0.8634   | 0.9091    | 0.8491 | 0.8780   | 🏆 Best |
| **KNN**                    | 0.8579   | 0.8846    | 0.8679 | 0.8762   | 2️⃣      |
| **Random Forest**          | 0.8579   | 0.8846    | 0.8679 | 0.8762   | 2️⃣      |
| **Sklearn Naive Bayes**    | 0.8470   | 0.9149    | 0.8113 | 0.8600   | 4️⃣      |
| **SVM**                    | 0.8306   | 0.8713    | 0.8302 | 0.8502   | 5️⃣      |
| **Sklearn Decision Tree**  | 0.8033   | 0.8977    | 0.7453 | 0.8144   | 6️⃣      |

---

## 🔍 Xem Metadata Models

### **1️⃣ File JSON (Đánh máy)**

```bash
cat MODEL_REGISTRY.json
```

Hoặc mở trong editor:

- VSCode: Mở `MODEL_REGISTRY.json` trực tiếp
- GitHub: Xem tại `MODEL_REGISTRY.json`

### **2️⃣ Python Script - Load Metadata**

```python
import json

with open('MODEL_REGISTRY.json', 'r') as f:
    registry = json.load(f)

# Xem thông tin từng model
for model in registry['models']:
    print(f"Model: {model['name']}")
    print(f"  F1-Score: {model['metrics']['f1_score']}")
    print(f"  Hyperparameters: {model['hyperparameters']}")
```

### **3️⃣ Terminal - Pretty Print JSON**

```bash
# Windows PowerShell
Get-Content MODEL_REGISTRY.json | ConvertFrom-Json | ConvertTo-Json

# Linux/Mac
python -m json.tool MODEL_REGISTRY.json
```

---

## 📁 Cấu Trúc Folder Models

```
models_output/
├── decision_tree_20260526_093507.pkl       ← Được sinh bởi scripts
├── decision_tree_20260526_094155.pkl       ← Tương tự
└── [các models mới được thêm tự động]

⚠️ Những file này KHÔNG được track trên GitHub
```

---

## ✅ Checklist - Khi Cần Dùng Models

- [ ] Clone repository
- [ ] Cài dependencies: `pip install -r requirements.txt`
- [ ] Generate models: `python main.py`
- [ ] Xem metadata: `cat MODEL_REGISTRY.json`
- [ ] Dùng model: `python scripts/predict.py --model "models_output/logistic_regression_*.pkl"`

---

## 💡 Lý Do Thiết Kế Này

| Vấn Đề                                     | Giải Pháp                          |
| ------------------------------------------ | ---------------------------------- |
| 🔴 Binary files không xem được trên GitHub | ✅ Dùng JSON metadata              |
| 🔴 Models quá nặng (3.7 KB mỗi cái)        | ✅ Generate locally khi cần        |
| 🔴 Không biết models được tạo khi nào      | ✅ Track trong MODEL_REGISTRY.json |
| 🔴 Không biết hyperparameters              | ✅ Documented in JSON              |

---

## 🎯 Best Practice

**Khi demo:**

1. Clone repo
2. Chạy `python main.py` (generate models + show results)
3. Khi muốn lưu model: `python scripts/train.py --model <name>`
4. Check metrics trong `MODEL_REGISTRY.json`

**Tương lai:**

- Có thể lưu models trên **MLflow, Weights & Biases, HuggingFace Model Hub**
- Hoặc tạo CI/CD để auto-train & upload models

---

**Happy Modeling!** 🎉
