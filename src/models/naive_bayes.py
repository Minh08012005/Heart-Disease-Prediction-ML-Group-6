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

        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = {c: counts[i] / n_samples for i, c in enumerate(self.classes)}

        # means and variances per class
        self.means = {}
        self.variances = {}
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            # add small epsilon to variance to avoid division by zero
            self.variances[c] = np.var(X_c, axis=0) + 1e-9
    
    def _gaussian_pdf(self, x, mean, var):
        """Gaussian probability density function (per feature)."""
        coef = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = - ((x - mean) ** 2) / (2.0 * var)
        return coef * np.exp(exponent)
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if len(self.classes) == 0:
            raise ValueError("Model chưa được fit! Hãy gọi fit() trước.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        predictions = []

        for i in range(n_samples):
            x = X[i]
            class_log_posteriors = []
            for c in self.classes:
                log_prob = np.log(self.priors[c])
                mean = self.means[c]
                var = self.variances[c]
                pdfs = self._gaussian_pdf(x, mean, var)
                pdfs = np.maximum(pdfs, 1e-12)
                log_prob += np.sum(np.log(pdfs))
                class_log_posteriors.append(log_prob)

            # choose class with highest log posterior
            best_index = np.argmax(class_log_posteriors)
            predictions.append(self.classes[best_index])

        return np.array(predictions)
    
    def predict_proba(self, X):
        """Return class probabilities for samples in X."""
        if len(self.classes) == 0:
            raise ValueError("Model chưa được fit! Hãy gọi fit() trước.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            x = X[i]
            log_post = np.zeros(n_classes)
            for idx, c in enumerate(self.classes):
                logp = np.log(self.priors[c])
                mean = self.means[c]
                var = self.variances[c]
                pdfs = self._gaussian_pdf(x, mean, var)
                pdfs = np.maximum(pdfs, 1e-12)
                logp += np.sum(np.log(pdfs))
                log_post[idx] = logp

            # softmax in log-space for numerical stability
            m = np.max(log_post)
            exp_vals = np.exp(log_post - m)
            probs[i, :] = exp_vals / np.sum(exp_vals)

        return probs
