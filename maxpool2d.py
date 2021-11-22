
import numpy as np
from layer import Layer
from util import pool2d
from numpy import hstack

class Maxpool2d(Layer):
    def __init__(self, padding, kernel_size, stride, output_shape):
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_shape = output_shape

    def __str__(self):
        return f'{self.__class__.__name__}({self.padding}, {self.kernel_size}, {self.stride}, {self.output_shape})'

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)
        for i in range(len(self.input)):
            self.output[i] = pool2d(input[i], self.kernel_size, self.stride, padding=self.padding).max(axis=(2, 3))
        return self.output

    def backward(self, output_gradient):
        input_gradient = np.zeros(self.input.shape)
        for i in range(len(self.input)):
            output = pool2d(self.input[i], self.kernel_size, self.stride, padding=self.padding)
            mapped = [
                (output[d1,d2] == output[d1,d2].max()) * output_gradient[i,d1,d2] \
                for d1 in range(len(output)) for d2 in range(len(output[d1]))
            ]
            mapped = np.array(mapped).astype('float')
            mapped = hstack(hstack(mapped.reshape(output.shape)))
            input_gradient[i] = mapped[self.padding:, self.padding:]
        return input_gradient