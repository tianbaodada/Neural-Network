import numpy as np
from scipy.signal.filter_design import EPSILON

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred + EPSILON) - (1 - y_true) * np.log(1 - y_pred + EPSILON))

def binary_cross_entropy_prime(y_true, y_pred):
    return np.nan_to_num(((1 - y_true) / (1 - y_pred + EPSILON) - y_true / (y_pred + EPSILON)) / np.size(y_true))
