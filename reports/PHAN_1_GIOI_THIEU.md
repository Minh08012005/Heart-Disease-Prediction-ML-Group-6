# PHẦN 1: GIỚI THIỆU

## 1.1 Bài toán dự đoán bệnh tim

Bệnh tim mạch (Cardiovascular Disease - CVD) là một trong những nguyên nhân gây tử vong hàng đầu trên thế giới. Theo Tổ chức Y tế Thế giới (WHO), ước tính có khoảng 17.9 triệu người chết mỗi năm do các bệnh tim mạch, chiếm 31% tổng số ca tử vong toàn cầu. Việc phát hiện sớm các dấu hiệu bệnh tim đóng vai trò quan trọng trong việc điều trị và giảm thiểu rủi ro.

Bài toán đặt ra: **Xây dựng mô hình Machine Learning có khả năng dự đoán nguy cơ mắc bệnh tim dựa trên các chỉ số y tế của bệnh nhân.** Đây là bài toán **phân loại nhị phân** (Binary Classification), với đầu ra là:

- **0**: Không mắc bệnh tim
- **1**: Có mắc bệnh tim

## 1.2 Dataset

Nhóm sử dụng bộ dữ liệu **Heart Failure Prediction Dataset** từ Kaggle (tác giả: fedesoriano).

**Thông tin dataset:**

- Số lượng mẫu: **918 bệnh nhân**
- Số lượng features: **11 features** (6 features số + 5 features phân loại)
- Target: **HeartDisease** (0 = Không bệnh, 1 = Có bệnh)

**Mô tả các features:**

| Feature        | Kiểu      | Mô tả                                          |
| :------------- | :-------- | :--------------------------------------------- |
| Age            | Số        | Tuổi bệnh nhân (năm)                           |
| Sex            | Phân loại | Giới tính (M: Nam, F: Nữ)                      |
| ChestPainType  | Phân loại | Loại đau ngực (ATA, NAP, ASY, TA)              |
| RestingBP      | Số        | Huyết áp lúc nghỉ (mmHg)                       |
| Cholesterol    | Số        | Cholesterol huyết thanh (mg/dL)                |
| FastingBS      | Số        | Đường huyết lúc đói (0: <120, 1: >120 mg/dL)   |
| RestingECG     | Phân loại | Kết quả điện tâm đồ lúc nghỉ (Normal, ST, LVH) |
| MaxHR          | Số        | Nhịp tim tối đa đạt được                       |
| ExerciseAngina | Phân loại | Đau ngực do gắng sức (Y: Có, N: Không)         |
| Oldpeak        | Số        | ST depression - chỉ số thiếu máu cơ tim        |
| ST_Slope       | Phân loại | Độ dốc đoạn ST (Up, Flat, Down)                |
| HeartDisease   | Target    | 0 = Không bệnh, 1 = Có bệnh tim                |

## 1.3 Mục tiêu dự án

1. **Phân tích và khám phá dữ liệu** (EDA) để hiểu rõ đặc điểm của dataset
2. **Xử lý và làm sạch dữ liệu** (xử lý giá trị 0 vô lý, chuẩn hóa, mã hóa)
3. **Trực quan hóa dữ liệu** để phát hiện patterns và feature importance
4. **Xây dựng 2 thuật toán từ đầu** (tự code):
   - **Decision Tree**: Dựa trên Entropy và Information Gain
   - **Gaussian Naive Bayes**: Dựa trên Bayes Theorem
5. **So sánh và đánh giá** các mô hình bằng các metrics: Accuracy, Precision, Recall, F1-Score
6. **So sánh với thư viện sklearn** để kiểm chứng kết quả
