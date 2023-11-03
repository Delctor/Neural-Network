
import numpy as np
from layer import *
from numba.experimental import jitclass
import numba as nb

spec = [
        ("nInputs", nb.uint64), 
        ("nOutputs", nb.uint64), 
        ("layers", nb.types.ListType(Layer.class_type.instance_type)), 
        ]


@jitclass(spec)
class NeuralNetwork:
    def __init__(self, nInputs, layers, activationFunctions):
        self.nInputs = nInputs
        self.nOutputs = layers[-1]
        size = [nInputs] + layers
        self.layers = nb.typed.List([Layer(size[i], size[i + 1], activationFunctions[i]) for i in range(len(layers))])#[Layer(size[i], size[i + 1], activationFunctions[i]) for i in range(len(layers))]
        
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
                
                if self.layers[-1].activationFunctionType != "none":
                    self.layers[-1].dz = (-2 * (Y[i] - a)) * self.layers[-1].da if self.layers[-1].activationFunctionType != "softmax" else np.dot((-2 * (Y[i] - a)), self.layers[-1].da)
                else:
                    self.layers[-1].dz = (-2 * (Y[i] - a))
                self.layers[-1].backwardLastLayer()
                nextLayer = self.layers[-1]
                # Backward
                for j in range(len(self.layers) - 2, -1, -1):
                    self.layers[j].backward(nextLayer)
                    nextLayer = self.layers[j]
                
                if ((i + 1) % bathSize) == 0:
                    for j in nb.prange(len(self.layers)):
                        self.layers[j].updateParameters(bathSize, learningRate)
                
                loss += ((Y[i] - self.layers[-1].a) ** 2).mean()
            loss /= len(X)
            if printLoss:
                print(loss)
            
    def predict(self, X):
        X = np.expand_dims(X, 1)
        YHat = np.empty((X.shape[0], self.nOutputs))
        for i in nb.prange(len(X)):
            a = X[i]
            # Forward
            for layer in self.layers:
                layer.forward(a)
                a = layer.a
            YHat[i] = a
        return YHat
    
    
