"""
preprocessing.py - Module tiền xử lý dữ liệu cho Heart Disease Prediction

Module này chịu trách nhiệm:
1. Load dữ liệu từ file CSV
2. Tách features (X) và target (y)
3. Chuẩn hóa dữ liệu số (StandardScaler)
4. Mã hóa dữ liệu phân loại (OneHotEncoder)
5. Kết hợp cả hai bằng ColumnTransformer

Tác giả: Minh (Leader)
"""

import sys
import io
# Cấu hình UTF-8 để hiển thị tiếng Việt có dấu trên Windows Console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(filepath='data/heart_cleaned.csv'):
    """
    Hàm load dữ liệu từ file CSV.
    
    Parameters:
    - filepath: đường dẫn đến file CSV (mặc định: data/heart_cleaned.csv)
    
    Returns:
    - df: DataFrame chứa toàn bộ dữ liệu
    """
    print(f"[LOAD] Đang load dữ liệu từ: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   -> Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"   -> Target HeartDisease: 0={sum(df['HeartDisease']==0)}, 1={sum(df['HeartDisease']==1)}")
    return df


def separate_features_target(df, target_column='HeartDisease'):
    """
    Tách features (X) và target (y).
    
    Parameters:
    - df: DataFrame đầu vào
    - target_column: tên cột mục tiêu (mặc định: HeartDisease)
    
    Returns:
    - X: DataFrame chứa các features
    - y: Series chứa target
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"   [FEATURES] X có {X.shape[1]} cột")
    print(f"   [TARGET] y = {y.name}")
    print(f"   [COLUMNS] Danh sách features: {X.columns.tolist()}")
    
    return X, y


def create_preprocessor():
    """
    Tạo pipeline tiền xử lý với ColumnTransformer.
    
    ColumnTransformer cho phép áp dụng các bước biến đổi khác nhau
    lên các cột khác nhau trong cùng một pipeline.
    
    Các cột SỐ (numeric_features):
    - Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
    → Được chuẩn hóa bằng StandardScaler
    
    Các cột PHÂN LOẠI (categorical_features):
    - Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
    → Được mã hóa bằng OneHotEncoder (drop='first' để tránh đa cộng tuyến)
    
    Returns:
    - preprocessor: ColumnTransformer đã cấu hình
    """
    
    # Định nghĩa các cột số - cần chuẩn hóa
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    
    # Định nghĩa các cột phân loại - cần one-hot encoding
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    # Tạo ColumnTransformer
    # - ('num', ...): xử lý cột số
    # - ('cat', ...): xử lý cột phân loại
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    print("   [PREPROCESSOR] Đã tạo preprocessor với:")
    print(f"      - StandardScaler cho {len(numeric_features)} cột số: {numeric_features}")
    print(f"      - OneHotEncoder cho {len(categorical_features)} cột phân loại: {categorical_features}")
    
    return preprocessor


def preprocess_data(X, preprocessor=None):
    """
    Thực hiện tiền xử lý dữ liệu.
    
    Parameters:
    - X: DataFrame chứa features
    - preprocessor: ColumnTransformer (nếu None, sẽ tạo mới)
    
    Returns:
    - X_processed: numpy array đã được tiền xử lý
    - preprocessor: ColumnTransformer đã fit (để dùng lại sau)
    - feature_names: tên các cột sau khi xử lý
    """
    
    # Tạo preprocessor nếu chưa có
    if preprocessor is None:
        preprocessor = create_preprocessor()
    
    print("\n[PROCESSING] Đang tiền xử lý dữ liệu...")
    
    # Fit-transform dữ liệu
    # - fit: học các tham số (mean, std cho StandardScaler; các category cho OneHotEncoder)
    # - transform: áp dụng biến đổi lên dữ liệu
    X_processed = preprocessor.fit_transform(X)
    
    # Lấy tên các cột sau khi xử lý
    feature_names = _get_feature_names(preprocessor)
    
    print(f"   [DONE] Kích thước sau preprocessing: {X_processed.shape[0]} dòng, {X_processed.shape[1]} cột")
    print(f"   [FEATURES] Danh sách features sau xử lý ({len(feature_names)} cột):")
    for i, name in enumerate(feature_names):
        print(f"      [{i}] {name}")
    
    return X_processed, preprocessor, feature_names


def save_processed_data(X_processed, y, feature_names, filepath='data/heart_preprocessed.csv'):
    """
    Lưu dữ liệu đã tiền xử lý thành file CSV.
    
    Hàm này tạo ra file CSV mà các thành viên khác trong team
    có thể đọc trực tiếp để test thuật toán mà không cần
    chạy lại preprocessing.
    
    Parameters:
    - X_processed: numpy array đã tiền xử lý
    - y: target (HeartDisease)
    - feature_names: tên các cột sau xử lý
    - filepath: đường dẫn file đầu ra (mặc định: data/heart_preprocessed.csv)
    """
    # Tạo DataFrame từ numpy array + tên cột
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    
    # Thêm cột target vào cuối
    df_processed['HeartDisease'] = y.values
    
    # Lưu ra file CSV
    df_processed.to_csv(filepath, index=False)
    
    print(f"\n   [SAVE] Đã lưu dữ liệu đã xử lý vào: {filepath}")
    print(f"   [SAVE] Kích thước: {df_processed.shape[0]} dòng, {df_processed.shape[1]} cột")
    print(f"   [SAVE] 15 cột features + 1 cột HeartDisease = {df_processed.shape[1]} cột")
    print(f"   [SAVE] Các bạn khác có thể đọc bằng: pd.read_csv('{filepath}')")


def _get_feature_names(preprocessor):
    """
    Lấy tên các cột sau khi preprocessing.
    Hàm này giúp chúng ta biết được ý nghĩa của từng cột sau khi biến đổi.
    
    Parameters:
    - preprocessor: ColumnTransformer đã fit
    
    Returns:
    - feature_names: list tên các cột
    """
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            # Cột số: giữ nguyên tên
            feature_names.extend(columns)
        elif name == 'cat':
            # Cột phân loại: lấy tên từ OneHotEncoder
            encoder = transformer
            if hasattr(encoder, 'get_feature_names_out'):
                # sklearn >= 1.0
                cat_names = encoder.get_feature_names_out(columns)
                feature_names.extend(cat_names)
            else:
                # sklearn < 1.0 (fallback)
                for i, col in enumerate(columns):
                    categories = encoder.categories_[i][1:]  # bỏ category đầu (drop='first')
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
    
    return feature_names


def run_preprocessing_pipeline(filepath='data/heart_cleaned.csv'):
    """
    Chạy toàn bộ pipeline tiền xử lý.
    Đây là hàm tổng hợp để chạy một lần tất cả các bước.
    
    Parameters:
    - filepath: đường dẫn đến file CSV
    
    Returns:
    - X_processed: numpy array đã tiền xử lý
    - y: target
    - preprocessor: ColumnTransformer đã fit
    - feature_names: tên các cột sau xử lý
    """
    print("=" * 60)
    print("  HEART DISEASE PREDICTION - PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Bước 1: Load dữ liệu
    print("\n  BƯỚC 1: LOAD DỮ LIỆU")
    print("-" * 40)
    df = load_data(filepath)
    
    # Bước 2: Tách features và target
    print("\n  BƯỚC 2: TÁCH FEATURES VÀ TARGET")
    print("-" * 40)
    X, y = separate_features_target(df)
    
    # Bước 3: Tạo preprocessor
    print("\n  BƯỚC 3: TẠO PREPROCESSOR")
    print("-" * 40)
    preprocessor = create_preprocessor()
    
    # Bước 4: Tiền xử lý
    print("\n  BƯỚC 4: TIỀN XỬ LÝ DỮ LIỆU")
    print("-" * 40)
    X_processed, preprocessor, feature_names = preprocess_data(X, preprocessor)
    
    # Bước 5: Lưu dữ liệu đã xử lý ra file CSV
    print("\n  BƯỚC 5: LƯU DỮ LIỆU ĐÃ XỬ LÝ")
    print("-" * 40)
    save_processed_data(X_processed, y, feature_names)
    
    # Kết quả
    print("\n" + "=" * 60)
    print("  HOÀN THÀNH PREPROCESSING!")
    print("=" * 60)
    print(f"   Dữ liệu gốc: {X.shape}")
    print(f"   Dữ liệu sau preprocessing: {X_processed.shape}")
    print(f"   Target: {y.value_counts().to_dict()}")
    print("=" * 60)
    
    return X_processed, y, preprocessor, feature_names


# ============================================================
# CHẠY THỬ KHI EXECUTE FILE NÀY TRỰC TIẾP
# ============================================================
if __name__ == "__main__":
    X_processed, y, preprocessor, feature_names = run_preprocessing_pipeline()
