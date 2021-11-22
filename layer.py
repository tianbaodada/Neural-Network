class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def __str__(self):
        return f'{self.__class__.__name__}()'
        
    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient):
        # TODO: update parameters and return input gradient
        pass

    def update(self, learning_rate):
        pass

    def save(self, id):
        pass

    def load(self, id):
        pass