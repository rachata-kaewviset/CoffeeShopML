import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.1, regularization=None):
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha
        self.regularization = regularization

    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        if self.regularization == 'l1':
            theta_best = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * np.eye(X_b.shape[1])).dot(X_b.T).dot(y)
        elif self.regularization == 'l2':
            theta_best = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * np.eye(X_b.shape[1])).dot(X_b.T).dot(y)
        else:
            theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return np.floor(abs(X_b.dot(np.hstack((self.intercept_, self.coef_)))))
