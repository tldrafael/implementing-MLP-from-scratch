import layers_builder
import numpy as np
import utils
import sys


class NeuralNet:
        
    def __init__(self,  layers_dim, batch_size):
        self.layers_dim = layers_dim
        self.layer_out_id = len(layers_dim) - 1
        self.batch_size = batch_size
        self.idx = None

        
        #import pdb; pdb.set_trace();
        self.net = layers_builder.net_constructer(self.layers_dim, self.batch_size)
        self.err_history = []
        self.ll_history = []
        self.mse = None
        self.ll = None

        self.data_X = None
        self.data_Y = None
        self.data_X_batch = None

    
    
    def compute_gradient_approximation(self, i, weight_shift=1e-4):
        W_shape = self.net[i].W.shape
        for j_w in np.arange(W_shape[1]):
            for i_w in np.arange(W_shape[0]):
                # shift to minus limit
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] - weight_shift
                self.feed_forward_NN()
                self.mean_squared_error()
                mse_shift_negative = self.mse

                # remove shift
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] + weight_shift

                # shift to plus limit
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] + weight_shift
                self.feed_forward_NN()
                self.mean_squared_error()
                mse_shift_positive = self.mse

                # remove shift
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] - weight_shift

                # approximate gradient
                self.net[i].ga[i_w, j_w] = (mse_shift_positive - mse_shift_negative)/(2*weight_shift)                    



    def check_gradient_computation(self):
        # update all gradient errors
        self.compute_gradients_errors()
        
        # now do the same manually
        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            self.compute_gradient_approximation(i)
            check = self.net[i].check_gradient_computation(atol=1e-4)
            
            if not check:
                sys.exit("Error in compute gradient from layer " + str(self.net[i].layer_id))
                
        print "Gradient Checking is Matching!"
    
    
    
    def back_propagate_error(self):
        # Two-stage process: 
        ## 1st, a distinguish to get the output layer error
        ## 2nd, a standard computation to get the hidden layers errors
        
        # output layer
        activation_error = -(self.Y[self.idx] - self.net[self.layer_out_id].a)

        activation_derivative = utils.fun_sigmoid_derivative(self.net[self.layer_out_id].a)
        self.net[self.layer_out_id].d = np.multiply(activation_error, activation_derivative)

        # hidden layers
        for i in np.arange(1, self.layer_out_id, dtype=int)[::-1]:
            # the (-) exclude the row with the pior error that was directed to bias
            d_next = self.net[i+1].d
            if self.net[i+1].bias: 
                d_next = d_next[1:]
            
            
            d_activation = self.net[i].W.transpose().dot(d_next)
            summation_derivative = utils.fun_sigmoid_derivative(self.net[i].a)
            self.net[i].d = np.multiply(d_activation, summation_derivative)
            
            
    
    def compute_gradients_errors(self):
        # update layer errors
        self.back_propagate_error()
        
        # hidden layers
        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            layer_cur_activ_vector = self.net[i].a
            layer_next_error_vector = self.net[i+1].d
            
            # if layer next (l+1) has bias, remove its error row
            if self.net[i+1].bias: 
                layer_next_error_vector = layer_next_error_vector[1:]
            
            self.net[i].g = layer_next_error_vector.dot(layer_cur_activ_vector.transpose()) 
            # normalize by batch size
            self.net[i].g = self.net[i].g / self.batch_size
            
    
    
    def update_weights(self, r, check_grad=False):
        # get gradient error for each weight
        self.compute_gradients_errors()

        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            self.net[i].update_weights(r)
       
    
    
    def feed_forward_NN(self):
        for i in np.arange(0, self.layer_out_id + 1, dtype=int):
            # the first layer receive the input
            if( i == 0 ):
                self.net[i].a[1:] = self.X[self.idx, :].transpose()
                continue
                
            self.net[i].z = self.net[i-1].W.dot(self.net[i-1].a)
            self.net[i].set_activation()
        


    # CG = Check Gradient
    def train(self, X, Y, r, iterations, shuffle=False, CG=True):
        self.X = X
        self.Y = Y
        
        # order to roll over the samples
        X_ids_order = np.arange(self.X.shape[0], dtype=int)
        if shuffle:
            np.random.shuffle(X_ids_order)

        # no. of iterations to get an epoch
        itr_to_epoch = self.X.shape[0] / self.batch_size
        j = 0
        for i in np.arange(iterations):
            self.idx = X_ids_order[(j*self.batch_size):((j+1)*self.batch_size)]

            # mark position into data chunks
            j = j + 1
            if j >= itr_to_epoch:
                j = 0

            
            self.feed_forward_NN()
            self.update_weights(r)
            
            if CG and i < 5 :
                self.check_gradient_computation()
                
                
    
    def predict(self, x_i):
        # n_predictions control the size of the matrix which is to work
        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, 1)

        n_predictions = x_i.shape[1]

        for i in np.arange(0, self.layer_out_id + 1, dtype=int):
            if( i == 0 ):
                self.net[i].a[1:, :n_predictions] = x_i
                continue
            
            self.net[i].z[:, :n_predictions] = self.net[i-1].W.dot(self.net[i-1].a[:, :n_predictions])
            self.net[i].set_activation()
        
        return self.net[self.layer_out_id].a[:, :n_predictions]
    
    
    
    def train_fitted(self):
        train_fitted = np.array(([]))
        for i in np.arange(self.X.shape[0], dtype=int):
            p = self.predict(self.X[i])
            p = np.round(p)
            train_fitted = np.append(train_fitted, p)
        
        return train_fitted
    

    
    def mean_squared_error(self):
        h = self.net[self.layer_out_id].a
        y = self.Y[self.idx]
        self.mse = 0.5*np.power(y - h, 2)
        # normalize by batch_size
        self.mse = np.sum(self.mse) / self.batch_size

        
        
    
    def log_likelihood(self):
        h = self.predict(self.X[self.idx, :])
        y = self.Y[self.idx]
        self.ll = y*log(h) + (1-y)*log(1-h)
        # normalize by batch_size
        self.ll = np.sum(self.ll) / self.batch_size

