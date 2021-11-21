
import numpy as np
from layer import Layer
from util import pool2d
from numpy import hstack

class Maxpool2d(Layer):
    def __init__(self, padding, kernel_size, stride):
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, input):
        self.input = input
        self.poolshape = (
            len(self.input),
            (input.shape[1] - self.kernel_size) // self.stride + 1 + self.padding * 2, 
            (input.shape[2] - self.kernel_size) // self.stride + 1 + self.padding * 2
        )
        self.output = np.zeros(self.poolshape)
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
            input_gradient[i] = hstack(hstack(mapped.reshape(output.shape)))

        return input_gradient