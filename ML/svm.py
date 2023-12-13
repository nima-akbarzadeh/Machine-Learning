from sklearn.svm import SVC, SVR
import numpy as np
from datasets import load_classification_data
from utilities import accuracy

kernel_set = ['linear', 'poly', 'rbf', 'sigmoid']
C_set = [1, 10, 100]

svr_regressor = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svc_classifier = SVC(kernel='rbf', C=1.0)

# Training is as follows
# svr_regressor.fit(X_train, y_train)
# svr_classifier.fit(X_train, y_train)


class SVM_selfdesigned:

    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.b = None
        self.w = None

    def fit(self, X, y):
        # Make sure the classification is a binary task
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


if __name__ == '__main__':
    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='iris', data_info=True)

    lr_set = [0.1, 0.01, 0.001, 0.0001, 0.0000000001]
    lr = lr_set[2]

    SVM_classifier1 = SVC(kernel='linear', C=1.0)
    SVM_classifier1.fit(X_train, y_train)
    predictions1 = SVM_classifier1.predict(X_test)
    accuracy1 = accuracy(predictions1, y_test)
    print(f"Accuracy is {accuracy1:0.2f} for SVC.")

    SVM_classifier2 = SVM_selfdesigned(lr=lr)
    SVM_classifier2.fit(X_train, y_train)
    predictions2 = SVM_classifier2.predict(X_test)
    accuracy2 = accuracy(predictions2, y_test)
    print(f"Accuracy is {accuracy2:0.2f} for SVM_selfdesigned.")
