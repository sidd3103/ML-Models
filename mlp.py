import numpy as np
import findMin
import utils

# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b) 
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights,())])

def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes)-1):
        W_size = layer_sizes[i+1] * layer_sizes[i]
        b_size = layer_sizes[i+1]
        
        W = np.reshape(weights_flat[counter:counter+W_size], (layer_sizes[i+1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter+b_size][None]
        counter += b_size

        weights.append((W,b))  
    return weights

def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) # per-colmumn max

class MLP():
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, lammy=1, 
        epochs=100, batch_size = 1000, lr = 0.001, momentum_coef = 0.9):
    
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = lr
        self.beta = momentum_coef

    def funObj(self, weights_flat, X, y):
        weights = unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        for W, b in weights:
            Z = X @ W.T + b
            R = np.exp(Z)
            X = R/(1+R)
            activations.append(X)

        yhat = Z
        
        if self.classification: # softmax- TODO: use logsumexp trick to avoid overflow
            tmp = np.sum(np.exp(yhat), axis=1)
            # f = -np.sum(yhat[y.astype(bool)] - np.log(tmp))
            f = -np.sum(yhat[y.astype(bool)] - log_sum_exp(yhat))
            grad = np.exp(yhat) / tmp[:,None] - y
        else:  # L2 loss
            f = 0.5*np.sum((yhat-y)**2)  
            grad = yhat-y # gradient for L2 loss

        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes)-2,0,-1):
            W, b = weights[i]
            grad = grad @ W
            grad = grad * (activations[i] * (1-activations[i])) # gradient of logistic loss
            grad_W = grad.T @ activations[i-1]
            grad_b = np.sum(grad,axis=0)

            g = [(grad_W, grad_b)] + g # insert to start of list

        g = flatten_weights(g)
        
        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat**2)
        g += self.lammy * weights_flat 
        
        return f, g

    def mini_batch_sgd(self, X, y, weights_flat):
        n = X.shape[0]

        weights_flat_old = np.random.randn(weights_flat.shape[0])
        weights_flat_current = weights_flat

        for e in range(self.epochs):
            data = np.hstack((X,y))
            np.random.shuffle(data)
            X = data[:,:-10]
            y = data[:,-10:]
            
            start = 0
            end = self.batch_size


            while start < n:
                X_mini = X[start:end]
                y_mini = y[start:end]
                f,g = self.funObj(weights_flat_current, X_mini, y_mini)
                weights_flat_new = weights_flat_current - (self.alpha*g) + self.beta*(weights_flat_current - weights_flat_old)
                weights_flat_old = weights_flat_current
                weights_flat_current = weights_flat_new
                start = end
                end += self.batch_size

        return weights_flat_current

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:,None]
            
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)

        # Run mini-batch sgd
        weights_flat = self.mini_batch_sgd(X,y,weights_flat)
        self.weights = unflatten_weights(weights_flat, self.layer_sizes)


    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
        if self.classification:
            return np.argmax(Z,axis=1)
        else:
            return Z




