import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.1, regularization=None):
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha
        self.regularization = regularization

    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        predictions = X_b.dot(np.hstack((self.intercept_, self.coef_)))
        predictions[predictions < 0] = 0
        return np.floor(predictions).astype(int)