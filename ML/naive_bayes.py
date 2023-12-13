import numpy as np
from datasets import load_classification_data
from utilities import accuracy, normal_pdf
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

continuous_features = [0, 1]  # Example: Assuming the first two features are continuous
discrete_features = [2]  # Example: Assuming the third feature is discrete
binary_features = [3]  # Example: Assuming the fourth feature is binary

naive_bayes_classifier = GaussianNB()
# naive_bayes_classifier.fit(X_train, y_train)

# Create a ColumnTransformer to apply different preprocessing to different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', 'passthrough', continuous_features),  # No transformation for continuous
        ('discrete', CategoricalNB(), discrete_features),  # Apply CategoricalNB for discrete
        ('binary', BernoulliNB(), binary_features)  # Apply BernoulliNB for binary
    ])

# Combine the preprocessor with the classifier (in this case, GaussianNB for the final prediction)
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', GaussianNB())])

# Initialize a HalvingGridSearchCV to find the best hyperparameters (you can customize this)
param_grid = {'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]}
grid_search = HalvingGridSearchCV(model, param_grid, cv=5)


# grid_search.fit(X_train, y_train)


class NaiveBayes_selfdesigned:

    def __init__(self):
        self._classes = None
        self._priors = None
        self._var = None
        self._mean = None

    def fit(self, X, y):
        # problem parameters
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []
        # Assumption: data generated from a normal distribution
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(normal_pdf(x, self._mean[idx], self._var[idx])))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]


if __name__ == '__main__':

    # Example 1
    X_train, X_test, y_train, y_test = load_classification_data(data_name='breast', data_info=True)

    # knn_regressor1 = KNeighborsRegressor(n_neighbors=n_neighbors)
    # knn_regressor1.fit(X_train, y_train)
    # predictions1 = knn_regressor1.predict(X_test)
    # accuracy1 = accuracy(predictions1, y_test)
    # print(f"Accuracy is {accuracy1:0.2f} for KNeighborsRegressor.")

    naivebayes_classifier2 = NaiveBayes_selfdesigned()
    naivebayes_classifier2.fit(X_train, y_train)
    predictions2 = naivebayes_classifier2.predict(X_test)
    accuracy2 = accuracy(predictions2, y_test)
    print(f"Accuracy is {accuracy2:0.2f} for NaiveBayes_selfdesigned.")
