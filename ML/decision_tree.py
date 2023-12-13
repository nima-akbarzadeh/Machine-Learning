from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from utilities import entropy, majority_vote, accuracy
import numpy as np
from datasets import load_classification_data

classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
# classifier.fit(X_train, y_train)
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=10)


# regressor.fit(X_train, y_train)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


def _split(X_column, split_thresh):
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs


def _information_gain(y, X_column, split_thresh):
    # parent loss
    parent_entropy = entropy(y)

    # generate split
    left_idxs, right_idxs = _split(X_column, split_thresh)

    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0

    # compute the weighted avg. of the loss for the children
    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
    child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

    # information gain is difference in loss before vs. after split
    ig = parent_entropy - child_entropy
    return ig


def _best_criteria(X, y, feat_idxs):
    best_gain = -1
    split_idx, split_thresh = None, None
    for feat_idx in feat_idxs:
        X_column = X[:, feat_idx]
        thresholds = np.unique(X_column)
        for threshold in thresholds:
            gain = _information_gain(y, X_column, threshold)
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_thresh = threshold

    return split_idx, split_thresh


class DecisionTreeClassifier_selfdesigned:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = majority_vote(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = _best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = _split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)


if __name__ == '__main__':
    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='breast', data_info=True)

    lr_set = [0.1, 0.01, 0.001, 0.0001, 0.0000000001]
    lr = lr_set[2]

    DTC_classifier1 = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
    DTC_classifier1.fit(X_train, y_train)
    predictions1 = DTC_classifier1.predict(X_test)
    accuracy1 = accuracy(predictions1, y_test)
    print(f"Accuracy is {accuracy1:0.2f} for DTC.")

    DTC_classifier2 = DecisionTreeClassifier_selfdesigned(max_depth=5, min_samples_split=10)
    DTC_classifier2.fit(X_train, y_train)
    predictions2 = DTC_classifier2.predict(X_test)
    accuracy2 = accuracy(predictions2, y_test)
    print(f"Accuracy is {accuracy2:0.2f} for DTC_selfdesigned.")
