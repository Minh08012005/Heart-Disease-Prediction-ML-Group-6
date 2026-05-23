# PHẦN 3: EDA & PREPROCESSING

## 3.1 Khám phá dữ liệu (EDA)

### 3.1.1 Tổng quan dữ liệu

Dataset có **918 mẫu** với **11 features** (6 số + 5 phân loại) và 1 target.

```python
import pandas as pd
df = pd.read_csv('data/heart.csv')
print("Shape:", df.shape)  # (918, 12)
print(df.info())
print(df.describe())
```

**Kết quả:**

- Kích thước: 918 dòng, 12 cột (11 features + 1 target)
- Không có giá trị thiếu (missing values)
- Target phân bố: 410 không bệnh (44.7%) vs 508 có bệnh (55.3%)

### 3.1.2 Phân bố target

```
HeartDisease
0    410 (44.7%)
1    508 (55.3%)
```

Dataset có sự cân bằng tương đối giữa 2 class, không bị mất cân bằng nghiêm trọng.

### 3.1.3 Phân tích theo giới tính

```
Sex         0    1    All
F         130   75    205
M         280  433    713
All       410  508    918
```

- Nam giới chiếm đa số (713/918 = 77.7%)
- Tỷ lệ mắc bệnh ở nam: 433/713 = 60.7%
- Tỷ lệ mắc bệnh ở nữ: 75/205 = 36.6%
- Nam giới có nguy cơ mắc bệnh tim cao hơn nữ giới

### 3.1.4 Phân tích theo loại đau ngực

```
ChestPainType    0    1    Tỷ lệ bệnh (%)
ASY             83  393    82.56%
ATA            182   64    26.02%
NAP            131   42    24.28%
TA              14    9    39.13%
```

- **ASY (Asymptomatic)**: 82.56% mắc bệnh → nguy hiểm nhất vì không có triệu chứng
- **ATA, NAP**: Tỷ lệ bệnh thấp (~25%)
- Đây là feature quan trọng để phân loại

## 3.2 Xử lý giá trị 0 vô lý

### 3.2.1 Phát hiện

Kiểm tra các cột có giá trị 0 không hợp lý về mặt y học:

```python
print("Cholesterol=0:", (df['Cholesterol']==0).sum())  # 172 giá trị
print("RestingBP=0:", (df['RestingBP']==0).sum())      # 1 giá trị
```

- **Cholesterol = 0**: 172 giá trị (18.7%) — không thể có cholesterol = 0
- **RestingBP = 0**: 1 giá trị — huyết áp không thể bằng 0

### 3.2.2 Xử lý Cholesterol = 0

Thay bằng median của nhóm tương ứng (có bệnh/không bệnh):

```python
median_0 = df[df['HeartDisease'] == 0]['Cholesterol'].median()
median_1 = df[df['HeartDisease'] == 1]['Cholesterol'].median()

df.loc[(df['Cholesterol'] == 0) & (df['HeartDisease'] == 0), 'Cholesterol'] = median_0
df.loc[(df['Cholesterol'] == 0) & (df['HeartDisease'] == 1), 'Cholesterol'] = median_1
```

**Kết quả:** 172 giá trị Cholesterol=0 đã được thay thế.

### 3.2.3 Xử lý RestingBP = 0

Thay bằng median chung:

```python
median_bp = df['RestingBP'].median()
df.loc[df['RestingBP'] == 0, 'RestingBP'] = median_bp
```

**Kết quả:** 1 giá trị RestingBP=0 đã được thay thế.

## 3.3 Tiền xử lý (Preprocessing)

### 3.3.1 Chiến lược xử lý

Sử dụng **ColumnTransformer** để kết hợp 2 bước:

1. **StandardScaler**: Chuẩn hóa 6 cột số
2. **OneHotEncoder**: Mã hóa 5 cột phân loại

### 3.3.2 StandardScaler

**Công thức:** $Z = \frac{x - \mu}{\sigma}$

Biến đổi dữ liệu về phân phối chuẩn với mean=0, std=1.

**Áp dụng cho:** Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak

### 3.3.3 OneHotEncoder

Biến mỗi giá trị phân loại thành cột 0/1 riêng biệt.

**Ví dụ:** ChestPainType có 4 giá trị (ATA, NAP, ASY, TA) → tạo 3 cột (bỏ ASY do drop='first')

**Áp dụng cho:** Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

### 3.3.4 Code thực hiện

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)
```

### 3.3.5 Kết quả

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
