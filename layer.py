
import numpy as np
from activationFunctions import *
from numba.experimental import jitclass
import numba as nb

spec = [
        ("weights", nb.float64[:, :]), 
        ("biases", nb.float64[:, :]), 
        ("dwAcumulator", nb.float64[:, :]), 
        ("dbAcumulator", nb.float64[:, :]), 
        ("x", nb.float64[:, :]), 
        ("a", nb.float64[:, :]), 
        ("da", nb.float64[:, :]), 
        ("dz", nb.float64[:, :]), 
        ("activationFunction", nb.float64[:, :](nb.float64[:, :], nb.float64[:, :], nb.b1).as_type()), 
        ("activationFunctionType", nb.types.unicode_type)
        ]


@jitclass(spec)
class Layer:
    def __init__(self, nInputs, nNeurons, activationFunction):
        
        self.activationFunctionType = activationFunction
        
        if activationFunction == "sigmoid":
            self.activationFunction = sigmoid
        elif activationFunction == "tanh":
            self.activationFunction = tanh
        elif activationFunction == "relu":
            self.activationFunction = relu
        elif activationFunction == "lrelu":
            self.activationFunction = lrelu
        elif activationFunction == "softmax":
            self.activationFunction = softmax
        
        self.weights = np.random.rand(nInputs, nNeurons)
        self.biases = np.random.rand(1, nNeurons)
        
        self.dwAcumulator = np.zeros((nInputs, nNeurons))
        self.dbAcumulator = np.zeros((1, nNeurons))
    
    def forward(self, x):
        self.x = x
        z = np.dot(x, self.weights) + self.biases
        self.a = self.activationFunction(z, z, False)
        self.da = self.activationFunction(self.a, z, True)
    
    def forwardPredict(self, x):
        z = np.dot(x, self.weights) + self.biases
        return self.activationFunction(z, z, False)
    
    def backwardLastLayer(self):
        self.dwAcumulator += np.dot(self.x.T, self.dz)
        self.dbAcumulator += self.dz
    
    def backward(self, nextLayer):
        self.dz = self.da * np.dot(nextLayer.dz, nextLayer.weights.T)
        
        self.dwAcumulator += np.dot(self.x.T, self.dz)
        self.dbAcumulator += self.dz
    
    def updateParameters(self, n, learningRate):
        self.weights -= (self.dwAcumulator / n) * learningRate
        self.biases -= (self.dbAcumulator / n) * learningRate
    
        self.dwAcumulator = np.zeros(self.weights.shape)
        self.dbAcumulator = np.zeros(self.biases.shape)

