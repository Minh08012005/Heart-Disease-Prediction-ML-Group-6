# PHẦN 7: KẾT LUẬN

## 7.1 Tổng kết dự án

Dự án "Dự đoán bệnh tim bằng Machine Learning" đã hoàn thành với các kết quả chính:

### 7.1.1 Những gì đã làm được

1. **EDA & Preprocessing (Minh)**
   - Phân tích và khám phá 918 mẫu dữ liệu bệnh nhân
   - Xử lý 172 giá trị Cholesterol=0 và 1 giá trị RestingBP=0 vô lý
   - Chuẩn hóa dữ liệu bằng StandardScaler và OneHotEncoder
   - Kết quả: 15 features sẵn sàng cho Machine Learning

2. **Trực quan hóa dữ liệu (Minh)**
   - Correlation Matrix: phát hiện ST_Slope_Flat (+0.43) và MaxHR (-0.39) là features quan trọng nhất
   - Histogram & Boxplot: so sánh phân bố dữ liệu giữa 2 nhóm bệnh
   - Feature Importance: phân loại 15 features theo mức độ quan trọng

3. **Decision Tree (Hiếu, Phong, Minh)**
   - Implement từ đầu: Node, Entropy, Information Gain, đệ quy xây cây
   - Tuning tham số: max_depth=10, min_samples_split=20
   - Kết quả: Accuracy 80.33%, Precision 86.46%, Recall 78.30%
   - K-Fold CV: Mean 81.94%, Std 3.94%

4. **Naive Bayes (Tuân)**
   _(Chờ cập nhật)_

5. **So sánh với sklearn (Minh)**
   - So sánh 8 models (2 custom + 6 sklearn)
   - Logistic Regression dẫn đầu: Accuracy 86.34%, Precision 90.91%
   - KNN & Random Forest có Recall cao nhất: 86.79%

### 7.1.2 Bảng tổng kết

| Hạng mục                 | Kết quả                        |
| :----------------------- | :----------------------------- |
| Dataset                  | 918 mẫu, 11 features           |
| Preprocessing            | 15 features (6 số + 9 one-hot) |
| Custom Decision Tree     | 80.33% Accuracy                |
| Custom Naive Bayes       | _(chờ cập nhật)_               |
| Model sklearn tốt nhất   | Logistic Regression (86.34%)   |
| Model có Recall cao nhất | KNN & Random Forest (86.79%)   |

## 7.2 Bài học kinh nghiệm

1. **Tự code thuật toán giúp hiểu sâu**: Việc tự implement Decision Tree và Naive Bayes từ đầu giúp các thành viên hiểu rõ bản chất toán học, không chỉ dùng thư viện có sẵn.

2. **Xử lý dữ liệu là bước quan trọng nhất**: Dữ liệu sạch và được chuẩn hóa tốt quyết định phần lớn chất lượng model.

3. **Tuning tham số cải thiện đáng kể**: Custom DT từ 76.50% lên 80.33% sau tuning.

4. **K-Fold Cross Validation đánh giá khách quan**: Accuracy 81.94% ± 3.94% cho thấy model ổn định.

5. **Trong y tế, Recall quan trọng hơn Accuracy**: Bỏ sót người bệnh (False Negative) nguy hiểm hơn chẩn đoán nhầm (False Positive).

## 7.3 Hướng phát triển

1. **Feature Engineering**: Tạo thêm features mới từ dữ liệu có sẵn
2. **Ensemble Methods**: Kết hợp nhiều models (Voting, Stacking)
3. **Deep Learning**: Thử nghiệm Neural Network
4. **Deployment**: Xây dựng web app demo cho bác sĩ nhập chỉ số và nhận kết quả dự đoán
5. **Thu thập thêm dữ liệu**: Dataset lớn hơn sẽ cải thiện độ chính xác
