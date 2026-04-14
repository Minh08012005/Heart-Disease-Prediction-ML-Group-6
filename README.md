# Heart-Disease-Prediction-ML-Group-6

# 🫀 Dự Án Học Máy: Xây Dựng Mô Hình Dự Đoán Bệnh Tim (Heart Disease Prediction)

## 📌 Giới thiệu chung
Dự án này là bài tập lớn môn Machine Learning, tập trung vào việc áp dụng các kiến thức hàn lâm vào thực tế. Mục tiêu của nhóm là xây dựng hệ thống phân loại khả năng mắc bệnh tim của bệnh nhân dựa trên các chỉ số y tế. 

Điểm đặc biệt của dự án: Thay vì sử dụng các thuật toán đóng gói sẵn (như `scikit-learn`), nhóm tập trung vào việc **tự xây dựng các thuật toán từ con số 0 (from scratch)** bằng toán học và logic lập trình cơ bản. Điều này giúp nhóm hiểu sâu sắc bản chất, cơ chế hoạt động và cách tối ưu hóa mô hình.

* **Bộ dữ liệu sử dụng:** Heart Failure Prediction Dataset (từ Kaggle).
* **Quy mô dữ liệu:** Khoảng 918 mẫu với 11 đặc trưng (Features) và 1 biến mục tiêu (Target).

---

## 👥 Thành viên nhóm & Phân công nhiệm vụ

| Họ và Tên | Vai trò | Trách nhiệm chính trong dự án |
| :--- | :--- | :--- |
| **Minh** | Team Leader | • Khám phá dữ liệu (EDA) & Tiền xử lý (Xử lý Missing value, Encoding, Scaling).<br>• Quản lý kiến trúc project, tích hợp các module code tại `main.py`.<br>• Đánh giá và so sánh kết quả mô hình. |
| **Tuân** | Algorithm Dev | • Nghiên cứu toán học và tự code module thuật toán **Naive Bayes** (Gaussian).<br>• Tối ưu hàm tính xác suất và đóng gói thành Class chuẩn. |
| **Hiếu** | Algorithm Dev | • Nghiên cứu và tự code module thuật toán **Decision Tree**.<br>• Phụ trách phần logic rẽ nhánh và xây dựng cấu trúc cây. |
| **Phong** | Algorithm Dev | • Cùng Hiếu code thuật toán **Decision Tree** (Pair-programming).<br>• Phụ trách tính toán độ vẩn đục thông tin (Entropy/Gini Impurity). |

---

## 📂 Cấu trúc thư mục (Project Structure)

Dự án được thiết kế theo hướng Module hóa để các thành viên có thể làm việc song song:

```text
Heart-Disease-Prediction/
├── data/                       # Chứa dữ liệu đầu vào
│   ├── heart.csv               # Dữ liệu thô gốc
│   └── heart_cleaned.csv       # Dữ liệu đã làm sạch (Input cho thuật toán)
├── notebooks/                  # Chứa file phân tích trực quan
│   └── 01_EDA_Preprocessing.ipynb
├── src/                        # Thư mục chứa mã nguồn chính
│   ├── models/                 # Chứa thuật toán tự code từ đầu
│   │   ├── naive_bayes.py      # Code của Tuân
│   │   └── decision_tree.py    # Code của Hiếu & Phong
│   └── utils.py                # Các hàm dùng chung (tính Accuracy, tính ma trận...)
├── .gitignore                  # Bỏ qua các file rác khi đẩy code lên GitHub
├── requirements.txt            # Danh sách thư viện môi trường
├── README.md                   # Tài liệu giới thiệu dự án
└── main.py                     # FILE CHẠY CHÍNH: Nạp dữ liệu và gọi các mô hình