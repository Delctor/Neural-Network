
import numpy as np
import numba as nb

@nb.njit
def sigmoid(a, z, derivate = False):
    if derivate:
        return a * (1 - a)
    else:
        return 1 / (1 + np.exp(-z))

@nb.njit
def tanh(a, z, derivate = False):
    if derivate:
        return 1 - a ** 2
    else:
        return np.tanh(z)
    
@nb.njit
def relu(a, z, derivate = False):
    if derivate:
        return np.where(a == 0.0, 0.0, 1.0)
    else:
        return np.maximum(0.0, z)

@nb.njit
def lrelu(a, z, derivate = False):
    if derivate:
        return np.where(z < 0.0, 0.01, 1)
    else:
        return np.where(z <= 0.0, z * 0.01, z)

@nb.njit
def diagflat(arr):
    result = np.empty((arr.shape[0], arr.shape[0]))
    
    for i in nb.prange(arr.shape[0]):
        for j in range(arr.shape[0]):
            if i == j:
                result[i, j] = arr[i, 0]
            else:
                result[i, j] = 0.0
    return result
    
@nb.njit
def softmax(a, z, derivate = False):
    if derivate:
        s = a.T
        return diagflat(s) - np.dot(s, s.T)
    else:
        exp = np.exp(z - np.max(z))
        return exp / exp.sum()
    
@nb.njit
def none(a, z, derivate = False):
    return z
    
