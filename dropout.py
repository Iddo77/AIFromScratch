import numpy as np

from layer import Layer


class Dropout(Layer):
    def __init__(self, dropout_rate: float):
        # A dropout rate of 0.2 - 0.5 is normal, but ReLu also sets a bunch of weights to 0.
        # So if you place the Dropout after a Relu, then the dropout would set even more weights to 0,
        # leading to neuron death. You can mitigate this, by using a smaller dropout rate.
        self.dropout_rate = dropout_rate
        self.train = True
        self.mask = []

    def forward(self, inputs) -> np.ndarray:
        if self.train:
            self.mask = np.random.rand(*inputs.shape) > self.dropout_rate
            # Apply the mask and scale the result to maintain expected value
            return inputs * self.mask / (1.0 - self.dropout_rate)
        else:
            # During evaluation, scale down uniformly
            return inputs * (1.0 - self.dropout_rate)

    def backward(self, gradients) -> np.ndarray:
        if self.train:
            return gradients * self.mask
        else:
            raise gradients
