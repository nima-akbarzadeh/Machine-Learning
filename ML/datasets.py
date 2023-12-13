from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_classification_data(data_name, data_info=True):
    if data_name == 'iris':
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1234)
        if data_info:
            print('------------ Data Info ------------')
            print(f"The X_train shape is {X_train.shape}")
            print(f"The y_train shape is {y_train.shape}")
            print(f"The X_test shape is {X_test.shape}")
            print(f"The y_test shape is {y_test.shape}")
            print('-----------------------------------')
            # plt.figure()
            # cmap = ListedColormap(['#FF0000', '##00FF00', '#0000FF'])
            # plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap=cmap, edgecolors='k', s=20)
            # plt.title("Distribution of the labels based on the first two features")
            # plt.show()
    elif data_name == 'breast':
        breast = datasets.load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=1234)
        if data_info:
            print('------------ Data Info ------------')
            print(f"The X_train shape is {X_train.shape}")
            print(f"The y_train shape is {y_train.shape}")
            print(f"The X_test shape is {X_test.shape}")
            print(f"The y_test shape is {y_test.shape}")
            print('-----------------------------------')
            # plt.figure()
            # plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap=cmap, edgecolors='k', s=20)
            # plt.title("Distribution of the labels based on the first two features")
            # plt.show()
    elif data_name == 'make':
        features, target = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1234)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1234)
        if data_info:
            print('------------ Data Info ------------')
            print(f"The X_train shape is {X_train.shape}")
            print(f"The y_train shape is {y_train.shape}")
            print(f"The X_test shape is {X_test.shape}")
            print(f"The y_test shape is {y_test.shape}")
            print('-----------------------------------')
            # plt.figure()
            # plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap=cmap, edgecolors='k', s=20)
            # plt.title("Distribution of the labels based on the first two features")
            # plt.show()
    else:
        X_train, X_test, y_train, y_test = None, None, None, None

    return X_train, X_test, y_train, y_test


def load_regression_data(data_name, data_info=True):
    X_train, X_test, y_train, y_test = None, None, None, None
    if data_name == 'make':
        features, target = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1234)
        if data_info:
            print('------------ Data Info ------------')
            print(f"The X_train shape is {X_train.shape}")
            print(f"The y_train shape is {y_train.shape}")
            print(f"The X_test shape is {X_test.shape}")
            print(f"The y_test shape is {y_test.shape}")
            print('-----------------------------------')
            # plt.figure()
            # plt.scatter(features[:, 0], target, color="b", marker="o", s=30)
            # plt.show()
    return X_train, X_test, y_train, y_test
