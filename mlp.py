
import numpy as np
from layer import *

class NeuralNetwork:
    def __init__(self, nInputs, layers, activationFunctions):
        self.nInputs = nInputs
        self.nOutputs = layers[-1]
        size = [nInputs] + layers
        self.layers = [Layer(size[i], size[i + 1], activationFunctions[i]) for i in range(len(layers))]
        
    def fit(self, X, Y, learningRate, epochs, bathSize, printLoss = False):
        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)
        for _ in range(epochs):
            loss = 0.0
            for i in range(len(X)):
                a = X[i]
                # Forward
                for layer in self.layers:
                    layer.forward(a)
                    a = layer.a
                
                self.layers[-1].dz = (-2 * (Y[i] - a)) * self.layers[-1].da if self.layers[-1].activationFunctionType != "softmax" else np.dot((-2 * (Y[i] - a)), self.layers[-1].da)
                nextLayer = None
                # Backward
                for layer in reversed(self.layers):
                    layer.backward(nextLayer)
                    nextLayer = layer

                if ((i + 1) % bathSize) == 0:
                    for layer in self.layers:
                        layer.updateParameters(len(X), learningRate)
                
                loss += (Y[i] - self.layers[-1].a) ** 2
            loss /= len(X)
            if printLoss:
                print(loss.mean())
            
    def predict(self, X):
        X = np.expand_dims(X, 1)
        YHat = np.empty((X.shape[0], self.nOutputs))
        for i in range(len(X)):
            a = X[i]
            # Forward
            for layer in self.layers:
                layer.forward(a)
                a = layer.a
            YHat[i] = a
        return YHat
        
        
        
        
        
        