import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, l2=0):
        self.weights = np.random.randn(output_size, input_size)
        # self.biases = np.ones((output_size, 1)) * 0.1
        self.biases = np.random.randn(output_size, 1)
        self.store_w = np.zeros(self.weights.shape)
        self.store_b = np.zeros(self.biases.shape)
        self.store_n = 0
        self.l2 = l2

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output

    def backward(self, output_gradient):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.store_w += weights_gradient
        self.store_b += output_gradient
        self.store_n += 1
        # self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * output_gradient
        return input_gradient

    def update(self, learning_rate):
        self.weights = (1 - self.l2 * learning_rate / self.store_n) * self.weights - (learning_rate / self.store_n) * self.store_w
        self.biases = self.biases - (learning_rate / self.store_n) * self.store_b
        self.store_w = np.zeros(self.weights.shape)
        self.store_b = np.zeros(self.biases.shape)
        self.store_n = 0

    def save(self, id):
        array_w = np.asarray(self.weights)
        array_b = np.asarray(self.biases)
        np.savez_compressed(f'save_data/{id}-weight.npz', array_w)
        np.savez_compressed(f'save_data/{id}-biases.npz', array_b)
    
    def load(self, id):
        array_w = np.load(f'save_data/{id}-weight.npz')
        array_b = np.load(f'save_data/{id}-biases.npz')
        self.weights = array_w['arr_0']
        self.biases = array_b['arr_0']