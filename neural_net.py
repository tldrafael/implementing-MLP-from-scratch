import layers_builder
import numpy as np
import sys

objective_options = {'linear-reg':'self.mean_squared_error', 'logistic-reg':'self.log_likelihood'}
def validate_objective_functions(obj):
    if obj not in objective_options.keys():
        sys.exit("the objective tasks available are %s" %(objective_options))

    return obj



class NeuralNet:
        
    def __init__(self,  layers_dim, batch_size, objective):
        self.objective = validate_objective_functions(objective)        
        self.objective_fun = eval(objective_options[self.objective])

        self.layers_dim = layers_dim
        self.layer_out_id = len(layers_dim) - 1
        self.batch_size = batch_size
        self.net = layers_builder.net_constructer(self.layers_dim, self.batch_size, self.objective)
        
        self.data_X = None
        self.data_Y = None
        self.idx = None
        self.data_X_batch = None
        
        self.obj_history = []
        
        
    
    def compute_gradient_approximation(self, i, weight_shift=1e-4):
        W_shape = self.net[i].W.shape
        for j_w in np.arange(W_shape[1]):
            for i_w in np.arange(W_shape[0]):
                # shift to minus limit
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] - weight_shift
                shift_negative = self.objective_fun()
                # remove shift
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] + weight_shift

                # shift to plus limit
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] + weight_shift
                shift_positive = self.objective_fun()
                # remove shift
                self.net[i].W[i_w, j_w] = self.net[i].W[i_w, j_w] - weight_shift

                # approximate gradient
                self.net[i].ga[i_w, j_w] = (shift_positive - shift_negative)/(2*weight_shift)                    



    def gradient_checking(self):
        # now do the same manually
        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            self.compute_gradient_approximation(i)
            check = self.net[i].check_gradient_computation(atol=1e-1)
            
            if not check:
                print "g:"
                print self.net[i].g
                print "\nga:"
                print self.net[i].ga
                sys.exit("Error in compute gradient from layer " + str(self.net[i].layer_id))
                
        print "Gradient Checking is Matching!"
    
    
    
    def back_propagate_error(self):
        # Two-stage process: 
        ## 1st, a distinguish to get the output layer error
        ## 2nd, a standard computation to get the hidden layers errors
        
        # output layer
        derv_cost_by_activation = -(self.data_Y[self.idx] - self.net[self.layer_out_id].a)
        derv_activation_by_summation = self.net[self.layer_out_id].get_derivative_of_activation()
        self.net[self.layer_out_id].d = np.multiply(derv_cost_by_activation, derv_activation_by_summation)


        # hidden layers
        for i in np.arange(1, self.layer_out_id, dtype=int)[::-1]:
            d_next = self.net[i+1].d
            if self.net[i+1].bias: 
                d_next = d_next[1:]
            
            derv_summation_lnext_by_activation = self.net[i].W.transpose().dot(d_next)
            derv_activation_by_summation = self.net[i].get_derivative_of_activation()
            self.net[i].d = np.multiply(derv_summation_lnext_by_activation, derv_activation_by_summation)
            
            
    
    def compute_gradients_errors(self):
        # update layer errors
        self.back_propagate_error()
        
        # hidden layers
        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            layer_cur_activations = self.net[i].a
            layer_next_errors = self.net[i+1].d
            
            # if layer next (l+1) has bias, remove its error row
            if self.net[i+1].bias: 
                layer_next_errors = layer_next_errors[1:]
            
            self.net[i].g = layer_next_errors.dot(layer_cur_activations.transpose()) 
            # normalize by batch size
            self.net[i].g = self.net[i].g / self.batch_size
            
    
    
    def update_weights(self, r, check_grad):
        # get gradient error for each weight
        self.compute_gradients_errors()

        if check_grad :
            self.gradient_checking()

        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            self.net[i].update_weights(r)
       
    
    
    def feed_forward_NN(self):
        for i in np.arange(0, self.layer_out_id + 1, dtype=int):
            # the first layer receive the input
            if( i == 0 ):
                self.net[i].a[1:] = self.data_X[self.idx, :].transpose()
                continue
                
            self.net[i].z = self.net[i-1].W.dot(self.net[i-1].a)
            self.net[i].set_activation()
        


    def train(self, X, Y, r, iterations, shuffle=False, check_grad=True):
        #import pdb; pdb.set_trace()
        self.data_X = X
        self.data_Y = Y

        # clean mse and ll past histories
        self.mse_history = []
        self.ll_history = []
        

        # order to roll over the samples
        data_X_ids_order = np.arange(self.data_X.shape[0], dtype=int)
        if shuffle:
            np.random.shuffle(data_X_ids_order)

        
        # no. of iterations to get an epoch
        itr_to_epoch = self.data_X.shape[0] / self.batch_size
        j = 0
        for i in np.arange(iterations):
            self.idx = data_X_ids_order[(j*self.batch_size):((j+1)*self.batch_size)]

            # mark position into data chunks
            j = j + 1
            # reset if reach the mark if reach the whole sample
            if j >= itr_to_epoch: j = 0

            self.feed_forward_NN()           
            # register the history of optimization parameters
            self.metric_register()

            self.update_weights(r, check_grad)

            # turn off gradient checking
            if i >= 4: check_grad = False
        
        # compute the optimization values from the last adjusted weights
        self.metric_register()


    
    def predict(self, x_i):
        # n_predictions control the size of the matrix which is to work
        # based on the number of samples sent to predict function
        x_i = x_i.transpose()
        
        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, 1)
        n_predictions = x_i.shape[1]

        # feed foward process
        for i in np.arange(0, self.layer_out_id + 1, dtype=int):
            if( i == 0 ):
                self.net[i].a[1:, :n_predictions] = x_i
                continue
            
            self.net[i].z[:, :n_predictions] = self.net[i-1].W.dot(self.net[i-1].a[:, :n_predictions])
            self.net[i].set_activation()
        
        predictions = self.net[self.layer_out_id].a[:, :n_predictions]
        return predictions
    
    
    
    def train_fitted(self):
        train_fitted = np.array(([]))
        for i in np.arange(self.data_X.shape[0], dtype=int):
            p = self.predict(self.data_X[i])
            train_fitted = np.append(train_fitted, p)
        
        return train_fitted
    

    
    def mean_squared_error(self):
        #h = self.net[self.layer_out_id].a
        h = self.predict(self.data_X[self.idx])
        y = self.data_Y[self.idx]
        mse = 0.5*np.power(y - h, 2)
        # normalize by batch_size
        mse = np.sum(mse) / self.batch_size
        return mse

        
        
    
    def log_likelihood(self):
        h = self.predict(self.data_X[self.idx])
        y = self.data_Y[self.idx]
        ll = y*np.log(h) + (1-y)*np.log(1-h)
        # normalize by batch_size
        ll = np.sum(ll) / self.batch_size

        # invert tz he signal to create the negative log-likelihood function
        ll = -1*ll
        return ll


    
    def metric_register(self):
        obj_value = self.objective_fun()
        self.obj_history.append(obj_value)
        