import sys
import numpy as np
import utils


# Layer class is a map function which vinculates the *m* units from the layer (l) and link each one to
# the *n* units from the layer (l+1)
# the Weight matrix is composed by:
#   + rows -> units from Layer(l+1)
#   + columns -> units from Layer (l)
#
class Layer:
    batch_size = None

    def __init__(self, n_units_current, n_units_next, bias, layer_id):
        self.layer_id = layer_id
        self.n_units_current = n_units_current
        self.n_units_next = n_units_next
        self.bias = bias

        # Summation vector
        self.z = self.initialize_vector((self.n_units_current, Layer.batch_size))

        # Activation vector
        # Inialize the vector and then set the activation function
        self.a = self.initialize_vector((self.n_units_current, Layer.batch_size))
        self.set_activation()

        # Weight matrix that connect units from current layer to next layer
        self.W = self.initialize_weights()

        # Delta-error vector
        self.d = self.initialize_vector((self.bias + self.n_units_current, Layer.batch_size))

        # Gradient error vector
        self.g = self.initialize_vector(self.W.shape)

        # Gradient approximation vector
        self.ga = self.initialize_vector(self.g.shape)

    def initialize_weights(self):
        # case there's none next layer is the output layer, also there's no Weight matrix
        if self.n_units_next is None:
            return np.array([])
        else:
            weights = np.random.randn(self.n_units_next * (self.bias + self.n_units_current))
            weights = weights.reshape(self.n_units_next, self.bias + self.n_units_current)
            return weights

    def initialize_vector(self, n_dimensions):
        return np.random.normal(size=n_dimensions)

    def set_activation(self):
        self.a = utils.fun_sigmoid(self.z)
        if self.bias:
            self.add_activation_bias()

    def add_activation_bias(self):
        if len(self.a.shape) == 1:
            self.a = np.vstack((1, self.a))
        else:
            self.a = np.vstack((np.ones(self.a.shape[1]), self.a))

    def get_derivative_of_activation(self):
        return utils.fun_sigmoid_derivative(self.a)

    def update_weights(self, r):
        self.W += -(r * self.g)

    def check_gradient_computation(self, atol):
        return np.allclose(self.g, self.ga, atol=atol)

    def print_layer(self):
        print("W:\n {} \n".format(self.W))
        print("z: {}".format(self.z))
        print("a: {}".format(self.a))
        print("d: {}".format(self.d))
        print("g: {}".format(self.g))


# The output layer is an exception case of the Layer class
# No summation vector
class LayerInput(Layer):
    def __init__(self, n_units_current, n_units_next=None, bias=True, layer_id=0):
        Layer.__init__(self, n_units_current, n_units_next, bias, layer_id)
        self.z = []


# The hidden layer must have a link between the current units to next units
class LayerHidden(Layer):
    def __init__(self, n_units_current, n_units_next, bias=True, layer_id=None):
        Layer.__init__(self, n_units_current, n_units_next, bias, layer_id)


# The layer output is an exception case of the Layer class
# No bias, and no Weight matrix
class LayerOutput(Layer):
    def __init__(self, n_units_current, layer_id):
        Layer.__init__(self, n_units_current, n_units_next=None, bias=False, layer_id=None)
        self.g = []
        self.ga = []


class ObjLinearRegression(LayerOutput):
    def __init__(self, n_units_current, layer_id):
        LayerOutput.__init__(self, n_units_current, layer_id)
        self.objective = 'linear-reg'

    def set_activation(self):
        self.a = self.z

    def get_derivative_of_activation(self):
        return np.ones(shape=self.a.shape)


class ObjLogisticRegression(LayerOutput):
    def __init__(self, n_units_current, layer_id):
        LayerOutput.__init__(self, n_units_current, layer_id)
        self.objective = 'logistic-reg'


def net_constructer(layers_dim, batch_size, objective):
    if len(layers_dim) < 2:
        sys.exit("Neural Net must have at least 2 layers")

    Layer.batch_size = batch_size
    net = []
    # First stage: create input and hidden layers
    for i in np.arange(len(layers_dim) - 1, dtype=int):
        if i == 0:
            new_layer = LayerInput(layers_dim[i], layers_dim[i + 1], bias=True)
            net.append(new_layer)
        else:
            new_layer = LayerHidden(layers_dim[i], layers_dim[i + 1], bias=True, layer_id=i)
            net.append(new_layer)

    # Second stage: create output layer
    if objective == 'linear-reg':
        new_layer = ObjLinearRegression(layers_dim[-1], layer_id=len(layers_dim))
    elif objective == 'logistic-reg':
        new_layer = ObjLogisticRegression(layers_dim[-1], layer_id=len(layers_dim))
    net.append(new_layer)

    return net
