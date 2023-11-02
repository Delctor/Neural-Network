
import numpy as np

def sigmoid(a, z, derivate = False):
    if derivate:
        return a * (1 - a)
    else:
        return 1 / (1 + np.exp(-z))

def tanh(a, z, derivate = False):
    if derivate:
        return 1 - a ** 2
    else:
        return np.tanh(z)
    
def relu(a, z, derivate = False):
    if derivate:
        return np.where(a == 0.0, 0.0, 1.0)
    else:
        return np.maximum(0.0, z)

def lrelu(a, z, derivate = False):
    if derivate:
        return np.where(z < 0.0, 0.01, 1)
    else:
        return np.where(z <= 0.0, z * 0.01, z)
    
def softmax(a, z, derivate = False):
    if derivate:
        s = a.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    else:
        exp = np.exp(z - np.max(z))
        return exp / exp.sum()
    