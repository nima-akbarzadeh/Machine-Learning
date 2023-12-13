import torch.nn as nn
import numpy as np
from datasets import load_classification_data
from utilities import accuracy, sigmoid


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


def logistic_regression(features, labels, num_iterations=1000, learning_rate=0.01):
    # Add a column of ones for the intercept term
    features = np.c_[np.ones(features.shape[0]), features]

    # Initialize weights
    weights = np.zeros(features.shape[1])

    for _ in range(num_iterations):
        # Compute predictions
        predictions = 1 / (1 + np.exp(-features @ weights))
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)  # Clip y_pred to avoid log(0) or log(1)
        # Compute BCE Loss
        error = - (labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
        # Update weights
        weights += learning_rate * features.T @ error

    return weights


class LogisticRegression_selfdesigned:
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
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
            dw = np.dot(X.T, (y_pred-y)) / n_samples
            db = np.sum(y_pred-y) / n_samples
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        predicted_values = sigmoid(np.dot(X_test, self.weights) + self.bias)
        predicted_labels = [1 if v > 0.5 else 0 for v in predicted_values]
        return predicted_values, predicted_labels


if __name__ == '__main__':

    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='breast', data_info=True)

    lr_set = [0.1, 0.01, 0.001, 0.0001]
    lr = lr_set[3]

    # knn_regressor1 = KNeighborsRegressor(n_neighbors=n_neighbors)
    # knn_regressor1.fit(X_train, y_train)
    # predictions1 = knn_regressor1.predict(X_test)
    # accuracy1 = accuracy(predictions1, y_test)
    # print(f"Accuracy is {accuracy1:0.2f} for KNeighborsRegressor.")

    logistic_regressor2 = LogisticRegression_selfdesigned(lr=lr)
    logistic_regressor2.fit(X_train, y_train)
    _, predictions2 = logistic_regressor2.predict(X_test)
    accuracy2 = accuracy(predictions2, y_test)
    print(f"Accuracy is {accuracy2:0.2f} for LogisticRegression_selfdesigned.")
