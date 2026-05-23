# PHẦN 6: SO SÁNH & ĐÁNH GIÁ

## 6.1 Bảng so sánh các models

Sau khi huấn luyện và đánh giá trên cùng bộ dữ liệu test (20% = 184 mẫu), ta thu được kết quả:

| Model                     |     Accuracy     |    Precision     |      Recall      |     F1-Score     |
| :------------------------ | :--------------: | :--------------: | :--------------: | :--------------: |
| **Custom Decision Tree**  |      80.33%      |      86.46%      |      78.30%      |      82.18%      |
| **Custom Naive Bayes**    | _(chờ cập nhật)_ | _(chờ cập nhật)_ | _(chờ cập nhật)_ | _(chờ cập nhật)_ |
| **Sklearn Decision Tree** |      82.51%      |      89.36%      |      79.25%      |      84.00%      |
| **Sklearn Naive Bayes**   | _(chờ cập nhật)_ | _(chờ cập nhật)_ | _(chờ cập nhật)_ | _(chờ cập nhật)_ |
| **SVM**                   |      83.06%      |      87.13%      |      83.02%      |      85.02%      |
| **KNN**                   |      85.79%      |      88.46%      |      86.79%      |      87.62%      |
| **Random Forest**         |      85.79%      |      88.46%      |      86.79%      |      87.62%      |
| **Logistic Regression**   |    **86.34%**    |    **90.91%**    |      84.91%      |    **87.80%**    |

## 6.2 Nhận xét

### 6.2.1 So sánh Custom vs Sklearn

**Decision Tree:**

- Custom DT: Accuracy 80.33% vs Sklearn DT: 82.51%
- Chênh lệch chỉ ~2% → code tự implement hoạt động tốt
- Sklearn có tối ưu thêm (pruning, split strategy) nên nhỉnh hơn

**Naive Bayes:**
_(Chờ cập nhật)_

### 6.2.2 So sánh giữa các thuật toán sklearn

1. **Logistic Regression (86.34%)** dẫn đầu về Accuracy và Precision (90.91%)
   - Phù hợp với dữ liệu có xu hướng tuyến tính
   - Precision cao → ít chẩn đoán nhầm người khỏe thành có bệnh

2. **KNN & Random Forest (85.79%)** có Recall cao nhất (86.79%)
   - Recall cao → phát hiện được nhiều người bệnh nhất
   - **Quan trọng trong y tế**: bỏ sót người bệnh (FN) nguy hiểm hơn chẩn đoán nhầm (FP)

3. **SVM (83.06%)** hoạt động tốt nhưng không nổi bật

4. **Custom Decision Tree (80.33%)** thấp nhất nhưng vẫn chấp nhận được

### 6.2.3 Lựa chọn model tốt nhất

**Tiêu chí y tế:** Recall là quan trọng nhất (phát hiện được người bệnh)

| Model                |   Recall   | Kết luận                             |
| :------------------- | :--------: | :----------------------------------- |
| KNN                  | **86.79%** | ✅ **Tốt nhất** - Phát hiện bệnh tốt |
| Random Forest        | **86.79%** | ✅ **Tốt nhất** - Phát hiện bệnh tốt |
| Logistic Regression  |   84.91%   | ✅ Tốt                               |
| SVM                  |   83.02%   | ✅ Tốt                               |
| Custom Decision Tree |   78.30%   | ⚠️ Chấp nhận được                    |

**Kết luận:** KNN và Random Forest là lựa chọn tốt nhất cho bài toán dự đoán bệnh tim vì có Recall cao nhất.

## 6.3 K-Fold Cross Validation

Để đánh giá khách quan hơn, nhóm thực hiện K-Fold Cross Validation (k=5) trên Custom Decision Tree:

| Fold     |  Accuracy  |
| :------- | :--------: |
| Fold 1   |   78.26%   |
| Fold 2   |   84.78%   |
| Fold 3   |   86.41%   |
| Fold 4   |   76.09%   |
| Fold 5   |   84.15%   |
| **Mean** | **81.94%** |
| **Std**  | **3.94%**  |

**Nhận xét:**

- Accuracy trung bình 81.94%, tương đương với kết quả trên test set (80.33%)
- Độ lệch chuẩn 3.94% → model ổn định, không bị overfitting
- Kết quả đáng tin cậy vì đã được đánh giá trên 5 folds khác nhau
