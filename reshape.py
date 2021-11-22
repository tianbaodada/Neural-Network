import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __str__(self):
        return f'{self.__class__.__name__}({self.input_shape}, {self.output_shape})'

    def forward(self, input):
        self.output = np.reshape(input, self.output_shape)
        return self.output

    def backward(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)
