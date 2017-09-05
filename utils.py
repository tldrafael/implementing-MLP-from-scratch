import numpy as np

def fun_sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def sigmoid_fun_derivative(a):
    derv = np.multiply(a, (1.0 - a))
    return derv


def add_bias(a_matrix):
    a_matrix = np.row_stack((np.ones(a_matrix.shape[1]), a_matrix))
    return a_matrix


def compute_mean_squared_error(a_3, idx):
    y_3 = y_norm[idx].reshape(a_3.shape)
    err = 0.5*np.power(y_3 - a_3, 2)

    err = np.sum(err)/batch_size
    return err


def log_likelihood(a_3, idx):
    y_3 = y_norm[idx].reshape(a_3.shape)

    ll = np.multiply(y_3, np.log(a_3)) + np.multiply(1 - y_3, np.log(1 - a_3))
    ll = np.sum(ll)/batch_size

    return ll

