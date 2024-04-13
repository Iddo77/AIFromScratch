import numpy as np

from layer import Layer


class Sequential(Layer):
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, inputs) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradients) -> np.ndarray:
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
        return gradients

    def params(self) -> list:
        return [param for layer in self.layers for param in layer.params()]

    def grads(self) -> list:
        return [grad for layer in self.layers for grad in layer.grads()]


