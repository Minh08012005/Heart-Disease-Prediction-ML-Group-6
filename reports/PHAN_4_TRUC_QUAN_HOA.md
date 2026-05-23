# PHẦN 4: TRỰC QUAN HÓA DỮ LIỆU

## 4.1 Correlation Matrix (Ma trận tương quan)

### 4.1.1 Khái niệm

Correlation (hệ số tương quan) đo lường mối quan hệ **tuyến tính** giữa 2 biến số, có giá trị từ **-1 đến +1**:

- **+1**: Tương quan thuận hoàn hảo (cùng tăng, cùng giảm)
- **0**: Không có tương quan tuyến tính
- **-1**: Tương quan nghịch hoàn hảo (cái tăng thì cái giảm)

### 4.1.2 Kết quả

Sau khi tính ma trận tương quan giữa 15 features và target HeartDisease, ta thu được:

**Các features tương quan mạnh nhất với HeartDisease:**

| Feature           | Correlation | Mức độ              |
| :---------------- | :---------: | :------------------ |
| ST_Slope_Flat     |    +0.43    | Mạnh                |
| ExerciseAngina_Y  |    +0.40    | Mạnh                |
| Oldpeak           |    +0.40    | Mạnh                |
| MaxHR             |    -0.39    | Mạnh (nghịch)       |
| ChestPainType_ATA |    -0.35    | Trung bình (nghịch) |
| Age               |    +0.28    | Trung bình          |
| Sex_M             |    +0.24    | Trung bình          |
| ST_Slope_Up       |    -0.24    | Trung bình (nghịch) |
| ChestPainType_NAP |    -0.16    | Yếu (nghịch)        |
| RestingECG_ST     |    +0.12    | Yếu                 |
| FastingBS         |    +0.11    | Yếu                 |
| RestingBP         |    +0.10    | Yếu                 |
| Cholesterol       |    +0.08    | Rất yếu             |
| RestingECG_Normal |    -0.07    | Rất yếu (nghịch)    |
| ChestPainType_TA  |    -0.02    | Không đáng kể       |

### 4.1.3 Nhận xét

- **ST_Slope_Flat** (độ dốc ST phẳng) là feature có tương quan mạnh nhất (+0.43)
- **MaxHR** có tương quan nghịch (-0.39): nhịp tim tối đa càng thấp, nguy cơ bệnh càng cao
- **Oldpeak** (+0.40): chỉ số thiếu máu cơ tim càng cao càng nguy hiểm
- **Cholesterol** (+0.08) và **RestingBP** (+0.10) có tương quan rất yếu với bệnh tim

## 4.2 Histogram & Boxplot

### 4.2.1 Histogram (Biểu đồ phân bố)

Histogram cho thấy phân bố của từng feature số theo 2 nhóm bệnh:

- **Age**: Nhóm có bệnh tập trung ở độ tuổi cao hơn (55-65) so với nhóm không bệnh (50-60)
- **MaxHR**: Nhóm có bệnh có nhịp tim tối đa thấp hơn rõ rệt → feature quan trọng
- **Oldpeak**: Nhóm có bệnh có Oldpeak cao hơn → chỉ số thiếu máu cơ tim cao
- **Cholesterol**: 2 nhóm phân bố gần như giống nhau → feature ít quan trọng
- **RestingBP**: Khác biệt không rõ rệt giữa 2 nhóm

### 4.2.2 Boxplot (Biểu đồ hộp)

Boxplot so sánh 5 số tóm tắt (Min, Q1, Median, Q3, Max) giữa 2 nhóm:

- **MaxHR**: Median của nhóm có bệnh (~120) thấp hơn nhóm không bệnh (~140)
- **Oldpeak**: Median của nhóm có bệnh (~0.8) cao hơn nhóm không bệnh (~0.0)
- **Age**: Median nhóm có bệnh (58) cao hơn nhóm không bệnh (52)
- **RestingBP, Cholesterol**: Hộp 2 nhóm chồng lấn nhiều → ít khác biệt

## 4.3 Feature Importance - Kết luận

### 4.3.1 Bảng tổng hợp

| Mức độ                | Features                                               | Số lượng |
| :-------------------- | :----------------------------------------------------- | :------- |
| ⭐⭐⭐ Rất quan trọng | ST_Slope_Flat, ExerciseAngina_Y, Oldpeak, MaxHR        | 4        |
| ⭐⭐ Quan trọng       | ChestPainType_ATA, Age, Sex_M, ST_Slope_Up             | 4        |
| ⭐ Ít quan trọng      | ChestPainType_NAP, RestingECG_ST, FastingBS, RestingBP | 4        |
| ❌ Không đáng kể      | Cholesterol, RestingECG_Normal, ChestPainType_TA       | 3        |

### 4.3.2 Những phát hiện chính

1. **Features quan trọng nhất** (nên giữ lại cho model):
   - `ST_Slope`: Chỉ số quan trọng nhất - độ dốc ST segment khi gắng sức
   - `ExerciseAngina`: Đau thắt ngực khi tập - yếu tố nguy cơ mạnh
   - `Oldpeak`: Chênh lệch ST segment - càng cao càng nguy hiểm
   - `MaxHR`: Nhịp tim tối đa - người bệnh thường có nhịp tim thấp hơn

2. **Features ít quan trọng** (có thể cân nhắc bỏ):
   - `RestingBP`: Huyết áp lúc nghỉ - khác biệt không rõ rệt
   - `FastingBS`: Đường huyết đói - ảnh hưởng yếu
   - `RestingECG`: Điện tâm đồ lúc nghỉ - ít tác dụng
   - `Cholesterol`: Tương quan rất yếu với bệnh tim

3. **Lưu ý:**
   - Giữ lại features có |correlation| > 0.1
   - Cẩn thận với multicollinearity (2 features quá giống nhau)
   - Kết hợp cả numeric và categorical features
