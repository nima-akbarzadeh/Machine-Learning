import numpy as np
from datasets import load_classification_data
from sklearn.decomposition import PCA


class PCA_selfdesigned:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # zero-mean data
        X = X - self.mean
        # project data
        return np.dot(X, self.components.T)


if __name__ == '__main__':
    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='breast', data_info=True)

    PCA_transformer1 = PCA(n_components=2)
    train_transformations = PCA_transformer1.transform(X_train)
    print(f"The new train data is")
    print(train_transformations[:5, :])
    test_transformations = PCA_transformer1.transform(X_test)
    print(f"The new test data is")
    print(test_transformations[:5, :])

    PCA_transformer2 = PCA_selfdesigned(n_components=2)
    PCA_transformer2.fit(X_train)
    train_transformations = PCA_transformer2.transform(X_train)
    print(f"The new train data is")
    print(train_transformations[:5, :])
    test_transformations = PCA_transformer2.transform(X_test)
    print(f"The new test data is")
    print(test_transformations[:5, :])

