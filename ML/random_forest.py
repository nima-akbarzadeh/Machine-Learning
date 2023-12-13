import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from decision_tree import DecisionTreeClassifier_selfdesigned
from utilities import bootstrap_sample, accuracy, majority_vote
from datasets import load_classification_data

classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10)
# classifier.fit(X_train, y_train)
regressor = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=10)
# regressor.fit(X_train, y_train)


class RandomForestClassifier_selfdesigned:

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier_selfdesigned(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Swap axis 0 and 1
        # Convert an array of format [1111 0000 2222] into [102 102 102 102]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [majority_vote(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)


if __name__ == '__main__':
    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='breast', data_info=True)

    RFC_classifier1 = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10)
    RFC_classifier1.fit(X_train, y_train)
    predictions1 = RFC_classifier1.predict(X_test)
    accuracy1 = accuracy(predictions1, y_test)
    print(f"Accuracy is {accuracy1:0.2f} for RFC.")

    RFC_classifier2 = RandomForestClassifier_selfdesigned(n_trees=10, max_depth=5, min_samples_split=10)
    RFC_classifier2.fit(X_train, y_train)
    predictions2 = RFC_classifier2.predict(X_test)
    accuracy2 = accuracy(predictions2, y_test)
    print(f"Accuracy is {accuracy2:0.2f} for RFC_selfdesigned.")
