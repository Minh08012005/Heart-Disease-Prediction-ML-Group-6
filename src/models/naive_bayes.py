"""Gaussian Naive Bayes classifier (Gaussian likelihood).

Simple from-scratch implementation with fit/predict/predict_proba.
"""

import numpy as np


class NaiveBayes:
    """Gaussian Naive Bayes classifier.

    Attributes:
    - priors: dict mapping class -> prior probability
    - means: dict mapping class -> feature means (1D array)
    - variances: dict mapping class -> feature variances (1D array)
    - classes: array of class labels (e.g. [0, 1])
    """

    def __init__(self):
        self.priors = {}
        self.means = {}
        self.variances = {}
        self.classes = np.array([])

    def fit(self, X, y):
        """Fit the model to data X and labels y."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Bước 1: Tính prior P(Class) cho mỗi class
        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = {c: counts[i] / n_samples for i, c in enumerate(self.classes)}

        # Bước 2: Tính mean và variance cho từng feature trong từng class
        self.means = {}
        self.variances = {}
        for c in self.classes:
            X_c = X[y == c]                                    # Lọc mẫu thuộc class c
            self.means[c] = np.mean(X_c, axis=0)                # Trung bình từng feature
            self.variances[c] = np.var(X_c, axis=0) + 1e-9     # Variance (+ epsilon tránh chia 0)

    def _gaussian_pdf(self, x, mean, var):
        """Gaussian probability density function P(x | mean, var).

        Công thức: f(x) = 1/sqrt(2*pi*var) * exp(-(x-mean)^2/(2*var))
        """
        coef = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = -((x - mean) ** 2) / (2.0 * var)
        return coef * np.exp(exponent)

    def _ensure_2d(self, X):
        """Đảm bảo X là mảng 2D (n_samples, n_features)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _predict_single(self, x):
        """Dự đoán class cho 1 mẫu x."""
        best_class = None
        best_log_prob = -np.inf

        for c in self.classes:
            # log(P(Class)) - log của prior
            log_prob = np.log(self.priors[c])

            # Cộng dồn log(P(feature_i | Class)) cho từng feature
            mean = self.means[c]
            var = self.variances[c]
            pdfs = self._gaussian_pdf(x, mean, var)
            pdfs = np.maximum(pdfs, 1e-300)          # Tránh log(0)
            log_prob += np.sum(np.log(pdfs))

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        return best_class

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters:
        - X: mảng features, shape (n_samples, n_features) hoặc (n_features,)

        Returns:
        - predictions: mảng labels, shape (n_samples,)
        """
        X = self._ensure_2d(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """Return class probabilities for samples in X.

        Parameters:
        - X: mảng features, shape (n_samples, n_features) hoặc (n_features,)

        Returns:
        - probs: ma trận xác suất, shape (n_samples, n_classes)
        """
        X = self._ensure_2d(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            x = X[i]
            log_posteriors = np.zeros(n_classes)

            for idx, c in enumerate(self.classes):
                log_prob = np.log(self.priors[c])
                mean = self.means[c]
                var = self.variances[c]
                pdfs = self._gaussian_pdf(x, mean, var)
                pdfs = np.maximum(pdfs, 1e-300)
                log_prob += np.sum(np.log(pdfs))
                log_posteriors[idx] = log_prob

            # Softmax in log-space để tránh underflow
            max_log = np.max(log_posteriors)
            exp_vals = np.exp(log_posteriors - max_log)
            probs[i, :] = exp_vals / np.sum(exp_vals)

        return probs