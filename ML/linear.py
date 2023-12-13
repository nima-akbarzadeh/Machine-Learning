import torch.nn as nn
import numpy as np
from datasets import load_regression_data
from utilities import mse_distance


class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x)

    # def fit(self, X, y):



def linear_regression_formula(features, labels):

    # Add bias term
    features = np.c_[np.ones(features.shape[0]), features]
    # Solution to MSE Loss
    weights = np.linalg.inv(features.T @ features) @ features.T @ labels

    return weights


class LinearRegression_selfdesigned:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # problem parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # training phase
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred-y) / n_samples
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        return np.dot(X_test, self.weights) + self.bias


if __name__ == '__main__':

    # Example 1
    X_train, X_test, y_train, y_test = load_regression_data(data_name='make', data_info=True)

    lr_set = [0.1, 0.01, 0.001, 0.0001]
    lr = lr_set[1]

    # linear_regressor1 = KNeighborsRegressor()
    # linear_regressor1.fit(X_train, y_train)
    # predictions1 = linear_regressor1.predict(X_test)
    # accuracy1 = 100 * np.sum(predictions1 == y_test) / len(y_test)
    # print(f"Accuracy is {accuracy1:0.2f} for KNeighborsRegressor.")

    linear_regressor2 = LinearRegression_selfdesigned(lr=lr)
    linear_regressor2.fit(X_train, y_train)
    predictions2 = linear_regressor2.predict(X_test)
    error2 = mse_distance(predictions2, y_test)
    print(f"MSE is {error2:0.4f} for LinearRegression_selfdesigned.")

