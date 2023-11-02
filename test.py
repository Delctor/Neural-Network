
import numpy as np
import mlp as MLP

# XOR

X = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

Y = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

mlp = MLP.NeuralNetwork(2, [4, 4, 2], ["tanh", "lrelu", "softmax"])

mlp.fit(X, Y, 0.2, 10000, 4, True)

mlp.predict(X)