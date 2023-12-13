import numpy as np
from datasets import load_classification_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA_selfdesigned:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * mean_diff.dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)


# Testing
if __name__ == "__main__":
    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='breast', data_info=True)

    # LDA_transformer1 = LinearDiscriminantAnalysis(n_components=2)
    # LDA_transformer1.fit(X_train, y_train)
    # train_transformations = LDA_transformer1.transform(X_train)
    # print(f"The new train data is")
    # print(train_transformations[:5, :])
    # test_transformations = LDA_transformer1.transform(X_test)
    # print(f"The new test data is")
    # print(test_transformations[:5, :])

    LDA_transformer2 = LDA_selfdesigned(n_components=2)
    LDA_transformer2.fit(X_train, y_train)
    train_transformations = LDA_transformer2.transform(X_train)
    print(f"The new train data is")
    print(train_transformations[:5, :])
    test_transformations = LDA_transformer2.transform(X_test)
    print(f"The new test data is")
    print(test_transformations[:5, :])
