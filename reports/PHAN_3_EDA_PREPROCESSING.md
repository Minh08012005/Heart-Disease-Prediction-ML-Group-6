# PHẦN 3: EDA & PREPROCESSING

## 3.1 Khám phá dữ liệu (EDA)

**Exploratory Data Analysis (EDA)** là quá trình phân tích dữ liệu ban đầu nhằm hiểu rõ cấu trúc, phân phối và các mối quan hệ giữa các đặc trưng thông qua thống kê mô tả và biểu đồ trực quan. EDA đóng vai trò định hướng cho các bước tiền xử lý và xây dựng mô hình ở giai đoạn tiếp theo.

### 3.1.1 Tổng quan dữ liệu

Dataset có **918 mẫu** với **11 features** (6 số + 5 phân loại) và 1 target.

| Chỉ tiêu               | Giá trị                 |
| :--------------------- | :---------------------- |
| Số mẫu                 | 918                     |
| Số features            | 11 (6 số + 5 phân loại) |
| Missing values         | 0                       |
| Target: 0 (Không bệnh) | 410 (44.7%)             |
| Target: 1 (Có bệnh)    | 508 (55.3%)             |

**Nhận xét:** Dataset cân bằng tương đối giữa 2 class, không bị mất cân bằng nghiêm trọng.

### 3.1.2 Phân tích theo giới tính

| Giới tính | Không bệnh | Có bệnh |  Tổng   | Tỷ lệ mắc bệnh |
| :-------- | :--------: | :-----: | :-----: | :------------: |
| Nữ (F)    |    143     |   50    |   193   |   **25.9%**    |
| Nam (M)   |    267     |   458   |   725   |   **63.2%**    |
| **Tổng**  |  **410**   | **508** | **918** |   **55.3%**    |

**Nhận xét:**

- Nam giới có nguy cơ mắc bệnh tim cao hơn đáng kể so với nữ giới (63.2% so với 25.9%), chênh lệch khoảng 2.4 lần. Điều này phù hợp với các nghiên cứu y khoa cho thấy nam giới có nguy cơ mắc bệnh tim mạch cao hơn nữ giới.
- Mẫu dữ liệu mất cân bằng về giới tính: Nam chiếm 79% tổng số mẫu. Đây là điểm cần lưu ý vì mô hình có thể học thiên về đặc điểm của nam giới.
- Feature Sex có khả năng phân loại tốt do sự khác biệt rõ rệt về tỷ lệ mắc bệnh giữa hai giới.

### 3.1.3 Phân tích theo loại đau ngực

**Giải thích 4 loại đau ngực:**

- **ASY (Asymptomatic)**: Không có triệu chứng — người bệnh không cảm thấy đau ngực
- **ATA (Atypical Angina)**: Đau ngực không điển hình — triệu chứng nhẹ, không rõ ràng
- **NAP (Non-Anginal Pain)**: Đau ngực không do tim — nguyên nhân từ cơ, xương, tiêu hóa
- **TA (Typical Angina)**: Đau thắt ngực điển hình — triệu chứng kinh điển của bệnh tim

| Loại đau ngực | Không bệnh | Có bệnh | Tổng | Tỷ lệ mắc bệnh |
| :------------ | :--------: | :-----: | :--: | :------------: |
| ASY           |    104     |   392   | 496  |   **79.03%**   |
| ATA           |    149     |   24    | 173  |   **13.87%**   |
| NAP           |    131     |   72    | 203  |   **35.47%**   |
| TA            |     26     |   20    |  46  |   **43.48%**   |

**Nhận xét:**

- **ASY** có tỷ lệ mắc bệnh cao nhất (79.03%) — đây là nhóm nguy hiểm nhất vì người bệnh không có triệu chứng cảnh báo nhưng thực tế đã mắc bệnh tim.
- **ATA** có tỷ lệ thấp nhất (13.87%) — phần lớn là người khỏe mạnh.
- **NAP** (35.47%) và **TA** (43.48%) ở mức trung bình.
- ChestPainType là feature quan trọng trong phân loại, đặc biệt ASY giúp nhận diện nhóm nguy cơ cao.

## 3.2 Tiền xử lý dữ liệu (Preprocessing)

### 3.2.1 Xử lý giá trị 0 vô lý (Data Cleaning)

Trong quá trình EDA, nhóm phát hiện các giá trị 0 không hợp lý về mặt y học:

| Cột         | Số giá trị 0 | Tỷ lệ | Giải thích                   |
| :---------- | :----------: | :---: | :--------------------------- |
| Cholesterol |     172      | 18.7% | Cholesterol không thể bằng 0 |
| RestingBP   |      1       | 0.1%  | Huyết áp không thể bằng 0    |

**Cách xử lý:**

- **Cholesterol = 0**: Thay bằng median theo nhóm bệnh (có bệnh/không bệnh) để bảo toàn đặc trưng phân bố của từng nhóm.
- **RestingBP = 0**: Thay bằng median chung.

**Nhận xét:** Nhóm sử dụng median thay vì mean vì median ít bị ảnh hưởng bởi outliers, đảm bảo dữ liệu sau xử lý vẫn phản ánh đúng phân bố thực tế.

### 3.2.2 Chuẩn hóa & Mã hóa (Feature Transformation)

Sử dụng **ColumnTransformer** để kết hợp 2 bước:

| Bước      | Phương pháp                  | Áp dụng cho                                              |
| :-------- | :--------------------------- | :------------------------------------------------------- |
| Chuẩn hóa | StandardScaler (Z-score)     | Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak   |
| Mã hóa    | OneHotEncoder (drop='first') | Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope |

**StandardScaler:** $Z = \frac{x - \mu}{\sigma}$ — biến đổi dữ liệu về phân phối chuẩn với mean=0, std=1.

**OneHotEncoder:** Biến mỗi giá trị phân loại thành cột 0/1 riêng biệt. Ví dụ: ChestPainType có 4 giá trị (ATA, NAP, ASY, TA) → tạo 3 cột (bỏ ASY do drop='first').

### 3.2.3 Kết quả

| Chỉ tiêu        | Trước xử lý        | Sau xử lý                 |
| :-------------- | :----------------- | :------------------------ |
| Số dòng         | 918                | 918                       |
| Số features     | 11                 | 15                        |
| Kiểu dữ liệu    | Hỗn hợp (số + chữ) | Toàn số                   |
| Phạm vi giá trị | Khác nhau          | Chuẩn hóa (mean=0, std=1) |

**15 cột features sau preprocessing:**

| Nhóm           | Số cột | Tên cột                                                |
| :------------- | :----- | :----------------------------------------------------- |
| Số (chuẩn hóa) | 6      | Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak |
| Sex            | 1      | Sex_M                                                  |
| ChestPainType  | 3      | ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA |
| RestingECG     | 2      | RestingECG_Normal, RestingECG_ST                       |
| ExerciseAngina | 1      | ExerciseAngina_Y                                       |
| ST_Slope       | 2      | ST_Slope_Flat, ST_Slope_Up                             |

Dữ liệu sau preprocessing được lưu vào `data/heart_preprocessed.csv`, sẵn sàng cho các thuật toán Machine Learning.
