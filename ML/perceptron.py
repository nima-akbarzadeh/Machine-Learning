import torch.nn as nn
import numpy as np
from datasets import load_classification_data
from utilities import accuracy, unit_step


class Perceptron(nn.Module):
    def __init__(self, input_size, activation_type):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Linear layer with one output unit
        self.act_type = activation_type
        if activation_type == 'Sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation_type == 'Tanh':
            self.act_fn = nn.Tanh()
        elif activation_type == 'SELU':
            self.act_fn = nn.SELU()
        elif activation_type == 'ReLU':
            self.act_fn = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        if self.act_type != 'None':
            out = self.act_fn(out)
        return out


class PerceptronClass_selfdesigned:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = unit_step
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # problem parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Make sure the classification is a binary task
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # training phase
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_pred = self.activation_func(np.dot(x_i, self.weights) + self.bias)
                update = self.lr * (y_[idx] - y_pred)
                self.weights -= update * x_i
                self.bias += update

    def predict(self, X_test):
        y_pred = self.activation_func(np.dot(X_test, self.weights) + self.bias)
        return y_pred


if __name__ == '__main__':
    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='iris', data_info=True)

    lr_set = [0.1, 0.01, 0.001, 0.0001, 0.0000000001]
    lr = lr_set[0]

    # knn_regressor1 = KNeighborsRegressor(n_neighbors=n_neighbors)
    # knn_regressor1.fit(X_train, y_train)
    # predictions1 = knn_regressor1.predict(X_test)
    # accuracy1 = accuracy(predictions1, y_test)
    # print(f"Accuracy is {accuracy1:0.2f} for KNeighborsRegressor.")

    Perceptron_classifier2 = PerceptronClass_selfdesigned(lr=lr)
    Perceptron_classifier2.fit(X_train, y_train)
    predictions2 = Perceptron_classifier2.predict(X_test)
    accuracy2 = accuracy(predictions2, y_test)
    print(f"Accuracy is {accuracy2:0.2f} for PerceptronClass_selfdesigned.")
