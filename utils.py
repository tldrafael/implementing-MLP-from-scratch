import numpy as np

def fun_sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def fun_sigmoid_derivative(a):
    derv = np.multiply(a, (1.0 - a))
    return derv
