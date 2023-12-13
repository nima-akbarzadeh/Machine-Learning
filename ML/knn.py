import numpy as np
from utilities import euclidean_distance, accuracy, majority_vote
from datasets import load_classification_data
from sklearn.neighbors import KNeighborsRegressor


class KNN_selfdesigned:
    def __init__(self, n_neighbors=3):
        self.X_train = None
        self.y_train = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest indices and labels
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_neighbors = [self.y_train[i] for i in k_indices]

        # majority vote as most common
        most_common = majority_vote(k_nearest_neighbors)

        return most_common


if __name__ == '__main__':

    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='iris', data_info=True)

    k_set = [1, 2, 3, 4, 5]
    n_neighbors = k_set[2]

    knn_regressor1 = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor1.fit(X_train, y_train)
    predictions1 = knn_regressor1.predict(X_test)
    accuracy1 = accuracy(predictions1, y_test)
    print(f"Accuracy is {accuracy1:0.2f} for KNeighborsRegressor.")

    knn_regressor2 = KNN_selfdesigned(n_neighbors=n_neighbors)
    knn_regressor2.fit(X_train, y_train)
    predictions2 = knn_regressor2.predict(X_test)
    accuracy2 = accuracy(predictions2, y_test)
    print(f"Accuracy is {accuracy2:0.2f} for KNN_selfdesigned.")
