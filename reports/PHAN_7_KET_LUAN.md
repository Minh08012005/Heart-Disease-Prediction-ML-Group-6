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
   - Kết quả: Accuracy 79.78%, Precision 87.91%, Recall 75.47%
   - K-Fold CV (Best params: max_depth=5, min_samples_split=2): Mean 83.12%, Std 1.43%

4. **Naive Bayes (Tuân)**
   - Implement từ đầu: Prior, Gaussian PDF, Log-Posterior
   - Kết quả: Accuracy 84.70%, Precision 91.49%, Recall 81.13%
   - Hoạt động chính xác ngang với thư viện sklearn

5. **So sánh với sklearn (Minh)**
   - So sánh 8 models (2 custom + 6 sklearn)
   - Logistic Regression dẫn đầu: Accuracy 86.34%, Precision 90.91%
   - KNN & Random Forest có Recall cao nhất: 86.79%

### 7.1.2 Bảng tổng kết

| Hạng mục                 | Kết quả                        |
| :----------------------- | :----------------------------- |
| Dataset                  | 918 mẫu, 11 features           |
| Preprocessing            | 15 features (6 số + 9 one-hot) |
| Custom Decision Tree     | 79.78% Accuracy (max_depth=10) |
| Custom Naive Bayes       | 84.70% Accuracy                |
| Model sklearn tốt nhất   | Logistic Regression (86.34%)   |
| Model có Recall cao nhất | KNN & Random Forest (86.79%)   |

## 7.2 Bài học kinh nghiệm

1. **Tự code thuật toán giúp hiểu sâu**: Việc tự implement Decision Tree và Naive Bayes từ đầu giúp các thành viên hiểu rõ bản chất toán học, không chỉ dùng thư viện có sẵn.

2. **Xử lý dữ liệu là bước quan trọng nhất**: Dữ liệu sạch và được chuẩn hóa tốt quyết định phần lớn chất lượng model.

3. **Tuning tham số cải thiện đáng kể**: Custom DT với best params (max_depth=5) đạt 82.51% Accuracy, cải thiện từ 79.78% (max_depth=10).

4. **K-Fold Cross Validation đánh giá khách quan**: Với best params, model đạt Accuracy 83.12% ± 1.43% - cho thấy model ổn định và không overfitting.

5. **Trong y tế, Recall quan trọng hơn Accuracy**: Bỏ sót người bệnh (False Negative) nguy hiểm hơn chẩn đoán nhầm (False Positive).

## 7.3 Hướng phát triển

1. **Feature Engineering**: Tạo thêm features mới từ dữ liệu có sẵn
2. **Ensemble Methods**: Kết hợp nhiều models (Voting, Stacking)
3. **Deep Learning**: Thử nghiệm Neural Network
4. **Deployment**: Xây dựng web app demo cho bác sĩ nhập chỉ số và nhận kết quả dự đoán
5. **Thu thập thêm dữ liệu**: Dataset lớn hơn sẽ cải thiện độ chính xác
