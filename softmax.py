import numpy as np
import utils
import findMin

# <x,y> = inner product of x and y

def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) 

class LogisticRegression:

	def __init__(self, maxEvals = 200, lammy = 1, penalty = 'L2'):
		self.maxEvals = maxEvals
		self.lammy = lammy
		self.penalty = penalty

	def funObj(self, w, X, y):
		n,d = X.shape
		W = np.reshape(w, (self.c,d))
		f = self.lammy/2 * np.sum(w**2) if self.penalty == 'L2' else 0 		# add regularization
		products = X@W.T 													# products[i,j] = <w_j, x_i>

		# Calculate Loss
		log_sum_exp_products = log_sum_exp(products)						# log_sum_exp_products[i] = sum_over_c(<w_c, x_i>)
		corrects = products[np.arange(n), y]								
		f += np.sum(log_sum_exp_products - corrects) 

		# calculate gradient
		Z = np.max(products, axis = 1)										# Z[i] = max_over_c(<w_c, x_i>)
		differences = products - Z[:,None]									# differences[i,j] = <w_j, x_i> - max_over_c(<w_c, x_i>)
		sum_exp_differences = np.exp(log_sum_exp(differences))
		I = np.zeros((n, self.c))											# I[i,j] = 1 if y_i = j, else 0
		I[np.arange(n),y] = 1
		expected_probs = np.exp(differences)/sum_exp_differences.reshape((-1,1))
		g = (expected_probs - I).T@X 

		if self.penalty == 'L2':
			g += self.lammy*W

		return f, g.flatten()

	def fit(self, X, y):
		n, d = X.shape
		# c = number of classes
		self.c = np.unique(y).size
		self.w = np.zeros(d*self.c)
		# utils.check_gradient(self, X, y)
		if self.penalty == 'L2':
			(self.w, f) = findMin.findMin(self.funObj, self.w,
                                      	self.maxEvals, X, y)
		else:
			(self.w, f) = findMin.findMinL1(self.funObj, self.w,self.lammy,
                                      	self.maxEvals, X, y)


	def predict(self, X):
		W = np.reshape(self.w, (self.c, X.shape[1]))
		return np.argmax(X@W.T, axis=1)