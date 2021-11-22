import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth, l2=0):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.weights = self.kernels
        # self.biases = np.ones(self.output_shape) * 0.1
        self.biases = np.random.randn(*self.output_shape)
        self.store_w = np.zeros(self.weights.shape)
        self.store_b = np.zeros(self.biases.shape)
        self.store_n = 0
        self.l2 = l2
    
    def __str__(self):
        return f'{self.__class__.__name__}({self.input_shape}, {self.kernel_size}, {self.depth}, l2={self.l2})'

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.weights[i, j], "valid")
        return self.output

    def backward(self, output_gradient):
        weights_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                weights_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.weights[i, j], "full")

        self.store_w += weights_gradient
        self.store_b += output_gradient
        self.store_n += 1
        # self.weights -= learning_rate * weights_gradient
        # self.biases -= learning_rate * output_gradient
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