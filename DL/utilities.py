import math
import sklearn
import numpy as np
from collections import Counter


def majority_vote(x):
    most_common_data = Counter(x).most_common(1)
    return most_common_data[0][0]


def euclidean_distance(x1, x2):
    if len(x1) != len(x1):
        raise Exception('The vectors must have the same size to calculate accuracy!')
    else:
        return math.dist(x1, x2)


def mse_distance(x1, x2):
    if len(x1) != len(x1):
        raise Exception('The vectors must have the same size to calculate accuracy!')
    else:
        return sklearn.metrics.mean_squared_error(x1, x2)


def accuracy(agent, oracle):
    if len(agent) != len(oracle):
        raise Exception('Agent and Oracle must have the same size to calculate accuracy!')
    else:
        return 100 * np.sum(agent == oracle) / len(oracle)


def sigmoid(x, temperature=1):
    return 1 / (1 + np.exp(-temperature * x))


def unit_step(x):
    return np.where(x >= 0, 1, 0)


def normal_pdf(x, mean, var):
    zero_indices = np.where(var == 0)
    if not zero_indices:
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    else:
        pdf = np.zeros(x.shape)
        for idx in range(len(x)):
            if idx in zero_indices:
                pdf[idx] = 1
            else:
                numerator = np.exp(- (x[idx] - mean[idx]) ** 2 / (2 * var[idx]))
                denominator = np.sqrt(2 * np.pi * var[idx])
                pdf[idx] = numerator / denominator
        return pdf


def entropy(x):
    hist = np.bincount(x)
    ps = hist / len(x)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]












