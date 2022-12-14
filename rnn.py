# library 
import numpy as np
import random 

# :::::::::::::::
# Activation functions:
# :::::::::::::::
# sigmoid function
def sigmoid(z):     
    return 1 / (1 + np.exp(-z))
# identity function
def identity(z):
    return z

# :::::::::::::::
# Objective function 
# :::::::::::::::
# mean square error (for single observation)
def loss_L2(target, pred): # L2 loss function
    return np.square(target - pred)
# derivate of l2 (for single observation)
def derivated_loss_L2(target, pred):    # L2 derivative function
    return 2 * (target - pred)
# mean square error for entire batch 
def total_loss(target, pred): 
    return np.sum(np.square(target - pred)) / pred.shape[0]



# :::::::::::::::
# Init network weights
# :::::::::::::::
def init_params(layer_dims):
    params = {}
    L = len(layer_dims)
    #
    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l])
        params['b'+str(l)] = np.random.randn(layer_dims[l], 1).T
    #
    return params




# :::::::::::::
# min-max features scaling 
# :::::::::::::
def scaling(x):
    x = x.astype(np.float64)
    for i in range(0,len(x[0])):
        mn = np.min(x[:,i])
        mx = np.max(x[:,i])
        scaled = np.array(list(map(lambda x: (x-mn)/(mx-mn), x[:,i])))
        x[:,i] = scaled
        
    return x


# :::::::::::::
# Convert pandas to numpy
# :::::::::::::
# da inserire scaling 
def setData(data, y_index, perc = 1, shuffle = False, scaler = False):
    
    # converto in numpy array
    if data.__class__.__name__ == 'DataFrame' or data.__class__.__name__ == 'list':
        data = np.array(data)
    elif data.__class__.__name__ == 'ndarray':
        pass
    else:
        raise TypeError('invalid class type for data')
        
    # Sep X (regression matrix) and y (response variables) 
    x = np.delete(data, y_index, axis=1)
    y = data[:,y_index]

    # control flow for train-test splitting process 
    if perc == 1:
        if scaler == False:
            return [x,y]
        else: 
            x = scaling(x)
            return [x,y]

    # split x and y in train and test 
    else:
        
        # scaling before shuffle
        if scaler == True:
            x = scaling(x)

        # shuffle phase 
        if shuffle == False:
            # set index 
            index = list(range(0,data.shape[0]))
            # define limit for splitting 
            limit = int((1-perc)*data.shape[0]) 
            train_index = index[0:limit]
            test_index = index[limit:]
            # train
            x_train = x[train_index,:]
            y_train = y[train_index]
            # test 
            x_test = x[test_index,:]
            y_test = y[test_index]
            # return matricies 
            return [x_train,y_train,x_test,y_test]
        else: 
            # set index
            index = list(range(0,data.shape[0]))
            # shuffle the index list 
            random.shuffle(index)
            # define limit for splitting
            limit = int((1-perc)*data.shape[0]) 
            train_index = index[0:limit]
            test_index = index[limit:]
            # train
            x_train = x[train_index,:]
            y_train = y[train_index]
            # test 
            x_test = x[test_index,:]
            y_test = y[test_index]
            # return matricies 
            return [x_train,y_train,x_test,y_test]


# :::::::::::::::
# Single hidden layer regression neural network
# :::::::::::::::
class rnn():
    def __init__(self, num_inputs, hidden_layer_size, sigmoid = sigmoid, identity = identity,
                 loss_L2 = loss_L2, derivated_loss_L2 = derivated_loss_L2, init_params = init_params, total_loss = total_loss):
        # definisco funzioni e quantit?? necessarie
        self.hidden_layer_size = hidden_layer_size
        self.num_inputs = num_inputs
        self.layer = [num_inputs, hidden_layer_size, 1]
        self.sigmoid = sigmoid
        self.identity = identity
        self.loss_L2 = loss_L2
        self.derivated_loss_L2 = derivated_loss_L2
        self.total_loss = total_loss
        self.params = init_params(self.layer)


    # :::::::::::::
    # forward
    # :::::::::::::
    def forward(self, x):
        # forward primo layer
        z_h = np.dot(x, self.params['W1']) + self.params['b1']
        a_h = sigmoid(z_h)
        # forward layer output
        z_o = np.dot(a_h, self.params['W2']) + self.params['b2']
        a_o = identity(z_o)
        nodes = [a_o, a_h, z_h]
        return nodes

    # :::::::::::::
    # forward - EVALUATE 
    # :::::::::::::
    def predict(self, x):
        z_h = np.dot(x, self.params['W1']) + self.params['b1']
        a_h = sigmoid(z_h)
        # forward layer output
        z_o = np.dot(a_h, self.params['W2']) + self.params['b2']
        a_o = identity(z_o)
        return a_o

    # :::::::::::::
    # Backward
    # :::::::::::::
    def backward(self, X, y, lr):
        for i in range(len(X)):
            f = self.forward(X[i])
            # :::::
            # bias output layer 
            # :::::
            dL_do = -derivated_loss_L2(y[i], f[0][0][0])
            do_db2 = 1
            dL_db2 = dL_do * do_db2
            # with the gradient descent update bias weight of the output node
            self.params['b2'] = self.params['b2'] - lr * dL_db2
            # return self.params['b2']
            # :::::
            # weights of the output layer 
            # :::::
            for j in range(len(self.params['W2'])):
                dL_do = -derivated_loss_L2(y[i], f[0][0][0])
                do_dW2_i = f[1][0][j]
                dL_dW2_i = dL_do * do_dW2_i 
                # aggiorno parametri 
                self.params['W2'][j] = self.params['W2'][j] - lr * dL_dW2_i
            # :::::
            # bias weights of the hidden layer 
            # :::::
            for k in range(len(self.params['b1'][0])): #(numero nodi strato latente)
                dL_do = -derivated_loss_L2(y[i], f[0][0][0])
                do_daok = self.params['W2'][k][0]
                daok_dzhk = sigmoid(f[2][0][k]) * (1 - sigmoid(f[2][0][k]))
                dzhk_dbk = 1 # perch?? nodo di bias non ha una feature associata 
                dL_dbk =  dL_do * do_daok * daok_dzhk * dzhk_dbk
                # aggiornamento parametri 
                self.params['b1'][0,k] = self.params['b1'][0,k] - lr * dL_dbk
            # :::::
            # weights of the hidden layer 
            # :::::
            # parametri primo nodo hidden layer 
            for k in range(len(self.params['b1'][0])):
                dL_do = -derivated_loss_L2(y[i], f[0][0][0])
                do_daok = self.params['W2'][k] 
                z = np.dot(X[i],self.params['W1'][:,k]) + self.params['b1'][0,k]
                daok_dzhk =  sigmoid(z) * (1 - sigmoid(z))
                for h in range(len(self.params['W1'][:,k])):
                    dzhk_dth = X[i,h]
                    # aggiornamento parametri 
                    dL_dth = dL_do * do_daok * daok_dzhk * dzhk_dth
                    self.params['W1'][h,k] = self.params['W1'][h,k] - lr * dL_dth


    # :::::::::::::
    # Train
    # :::::::::::::
    def train(self, X, y, lr, epochs):
        losses = []
        # Loop training
        for e in range(epochs):
            self.backward(X, y, lr)
            fwp = self.predict(X)
            loss = total_loss(y, fwp)
            losses.append(loss)
            print("Epoch {:4d}: training loss = {:.6f}".format(e, losses[e]))
 

                      
                 
# :::::::::  
# Test
# :::::::::        
if __name__ == "__main__":
    print('----------------------------------------------------')
    print('Test the training of single layer regressione neural netowrk')
    print('----------------------------------------------------')
    #########################
    # Activate the class instance
    model = rnn(num_inputs = 4, hidden_layer_size = 5)
    #########################
    # Create virtual training dataset  
    # Training test
    np.random.seed(610)
    # dataset 
    x = np.random.rand(8).reshape(2, 4)
    print('virtual dataset: \n')
    print(x)
    print('----------------------------------------------------')
    #########################
    # Create virtual target values
    y = np.array([[-2],[-1.5]])
    # iperparameters values selected 
    lr = 0.1
    #########################
    print('\n')
    print('::Training phase::')
    # launch the training for the network
    model.train(X = x, y = y, lr = lr, epochs = 200)
    #########################
    print('----------------------------------------------------')
    print('::Prediction test::')
    print('target value: ' + str(y[1]))
    print('net predicted value: ' + str(model.predict(x = x[1])) + '\n')