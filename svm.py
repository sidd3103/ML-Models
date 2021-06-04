import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class SVM:
	def __init__(self, lammy = 1, maxEvals = 100):
		self.maxEvals = maxEvals
		self.lammy = 1

	def loss(self,W,X,y,n):
		products = X@W.T										# products[i,j] = <x_i, w_j> 
		correct_one_d = products[np.arange(n),y].reshape(-1,1)	# correct_one_d[i] = <x_i, w_{y_i}> 
		corrects = np.ones(products.shape) * correct_one_d		# corrects[i,j] = <x_i, w_j> - <x_i, w_{y_i}>

		f = products - corrects + 1
		f[f < 0] = 0
		f[np.arange(n),y] = 0
		f = np.sum(f) + self.lammy/2 * np.sum(W**2)
		return f

	# Taken from : https://cs231n.github.io/optimization-1/
	def grad(self,W,X,y,n):
		products = X@W.T										# products[i,j] = <x_i, w_j>
		correct_one_d = products[np.arange(n),y].reshape(-1,1)	# correct_one_d[i] = <x_i, w_{y_i}> 
		corrects = np.ones(products.shape) * correct_one_d		# corrects[i,j] = <x_i, w_j> - <x_i, w_{y_i}> 
		f = products - corrects + 1 
		f[f < 0] = 0
		f[f > 0] = 1
		f[np.arange(n),y] = 0
		f[np.arange(n),y] = -np.sum(f,axis = 1)
		g = f.T@X + self.lammy * W
		return g


	def funObj(self, w, X, y):
		n,d = X.shape
		W = np.reshape(w, (self.k, d))
		
		f = self.loss(W,X,y,n)
		g = self.grad(W,X,y,n)
		return f,g.flatten()


	def fit(self, X,y):
		n,d = X.shape
		self.k = np.unique(y).size
		self.w = np.zeros(d*self.k)
		# utils.check_gradient(self,X,y)
		self.w,f = findMin.findMin(self.funObj, self.w, self.maxEvals, X, y)


	def predict(self, X):
		W = np.reshape(self.w, (self.k, X.shape[1]))
		return np.argmax(X@W.T, axis = 1)