import numpy as np
from layer import Layer
from activation import Activation

class Relu(Activation):
    def __init__(self, threshold = 0):
        def relu(x):
            x[x < 0] *= 0.01
            if threshold > 0:
                x[x > threshold] = 0.01 * threshold
            return x

        def relu_prime(x):
            x[x < 0] = 0.01
            x[x > 0] = 1
            if threshold > 0:
                x[x > threshold] = 0.01
            return x

        super().__init__(relu, relu_prime)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.nan_to_num(np.exp(input))
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient):
        n = np.size(self.output)
        tmp = np.tile(self.output, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)