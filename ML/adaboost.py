import numpy as np
from utilities import bootstrap_sample, accuracy, majority_vote
from datasets import load_classification_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor


# Decision stump used as weak classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class AdaboostClassifier_selfdesigned:

    def __init__(self, n_learners=5):
        self.n_learners = n_learners
        self.learners = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.learners = []

        # Iterate through classifiers
        for _ in range(self.n_learners):
            learner = DecisionStump()
            min_error = float("inf")

            # greedy search to find the best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        learner.polarity = p
                        learner.threshold = threshold
                        learner.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = 1e-10
            learner.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = learner.predict(X)

            w *= np.exp(-learner.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.learners.append(learner)

    def predict(self, X):
        learner_preds = [learner.alpha * learner.predict(X) for learner in self.learners]
        y_pred = np.sum(learner_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


# Testing
if __name__ == "__main__":
    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='breast', data_info=True)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    # Adaboost classification with 5 weak classifiers
    Adaboost2 = AdaboostClassifier_selfdesigned(n_learners=5)
    Adaboost2.fit(X_train, y_train)
    y_pred = Adaboost2.predict(X_test)

    accuracy2 = accuracy(y_test, y_pred)
    print(f"Accuracy is {accuracy2:0.2f} for AdaboostClassifier_selfdesigned.")
