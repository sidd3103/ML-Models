import os.path
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.optimize import approx_fprime
import pickle


def savefig(fname, verbose=True):
    plt.tight_layout()
    path = os.path.join('..', 'figs', fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma

def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.
    """

    # add extra dimensions so that the function still works for X and/or Xtest are 1-D arrays.
    if X.ndim == 1:
        X = X[None]
    if Xtest.ndim == 1:
        Xtest = Xtest[None]

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)


def check_gradient(model, X, y, verbose=True, epsilon=1e-6):
    # This checks that the gradient implementation is correct
    w = np.random.randn(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=epsilon)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient))/np.linalg.norm(estimated_gradient) > 1e-6:
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        if verbose:
            print('User and numerical derivatives agree.')
