import numpy as np


def fun_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fun_sigmoid_derivative(a):
    return np.multiply(a, (1.0 - a))
