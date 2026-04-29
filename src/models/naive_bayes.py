"""
naive_bayes.py - Module thuật toán Naive Bayes (Gaussian)

Tác giả: Tuân
Mô tả: Implement Gaussian Naive Bayes từ đầu dựa trên công thức Bayes:
    P(y|X) ∝ P(X|y) * P(y)
    
    Với giả định "naive": các features độc lập với nhau.
    Sử dụng Gaussian PDF để tính likelihood: 
    P(x_i|y) = 1/sqrt(2*pi*σ²) * exp(-(x_i-μ)²/(2σ²))
"""

import numpy as np

class NaiveBayes:
    """
    Gaussian Naive Bayes Classifier
    
    Attributes:
    - priors: dict, xác suất tiên nghiệm P(y) cho mỗi class
    - means: dict, giá trị trung bình của mỗi feature cho mỗi class
    - variances: dict, phương sai của mỗi feature cho mỗi class
    """
    
    def __init__(self):
        self.priors = {}
        self.means = {}
        self.variances = {}
        self.classes = None
    
    def fit(self, X, y):
        """
        Huấn luyện mô hình Naive Bayes.
        
        Parameters:
        - X: numpy array, ma trận features (n_samples, n_features)
        - y: numpy array, nhãn (n_samples,)
        """
        # TODO: Implement from scratch
        # 1. Xác định các class duy nhất
        # 2. Tính prior P(y) cho mỗi class
        # 3. Tính mean và variance cho mỗi feature trong mỗi class
        pass
    
    def _gaussian_pdf(self, x, mean, var):
        """
        Tính Probability Density Function của phân phối Gaussian.
        
        Công thức: f(x) = 1/sqrt(2*pi*σ²) * exp(-(x-μ)²/(2σ²))
        
        Parameters:
        - x: giá trị cần tính
        - mean: giá trị trung bình μ
        - var: phương sai σ²
        
        Returns:
        - probability: xác suất P(x|y)
        """
        # TODO: Implement from scratch
        pass
    
    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới.
        
        Parameters:
        - X: numpy array, ma trận features cần dự đoán
        
        Returns:
        - predictions: numpy array, nhãn dự đoán
        """
        # TODO: Implement from scratch
        # 1. Với mỗi sample, tính posterior log P(y|X) cho mỗi class
        # 2. Chọn class có posterior lớn nhất
        pass
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất cho mỗi class.
        
        Parameters:
        - X: numpy array, ma trận features cần dự đoán
        
        Returns:
        - probabilities: numpy array, xác suất cho mỗi class
        """
        # TODO: Implement from scratch
        pass
