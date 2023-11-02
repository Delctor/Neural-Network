
import numpy as np
from activationFunctions import *

class Layer:
    def __init__(self, nInputs, nNeurons, activationFunction):
        exec("self.activationFunction = " + activationFunction)
        self.activationFunctionType = activationFunction
        self.weights = np.random.rand(nInputs, nNeurons)
        self.biases = np.random.rand(1, nNeurons)
        
        self.dwAcumulator = np.zeros((nInputs, nNeurons))
        self.dbAcumulator = np.zeros((1, nNeurons))
    
    def forward(self, x):
        self.x = x
        z = np.dot(x, self.weights) + self.biases
        self.a = self.activationFunction(None, z, False)
        self.da = self.activationFunction(self.a, z, True)
    
    def forwardPredict(self, x):
        z = np.dot(x, self.weights) + self.biases
        return self.activationFunction(None, z, False)
    
    def backward(self, nextLayer = None):
        if nextLayer is not None:
            self.dz = self.da * np.dot(nextLayer.dz, nextLayer.weights.T)
        
        self.dwAcumulator += np.dot(self.x.T, self.dz)
        self.dbAcumulator += self.dz
    
    def updateParameters(self, n, learningRate):
        self.weights -= (self.dwAcumulator / n) * learningRate
        self.biases -= (self.dbAcumulator / n) * learningRate
    
        self.dwAcumulator = np.zeros(self.weights.shape)
        self.dbAcumulator = np.zeros(self.biases.shape)

