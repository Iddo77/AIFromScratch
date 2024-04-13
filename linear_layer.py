import numpy as np

from init_weights import WeightInitType, initialize_weights
from layer import Layer


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, weight_init_type=WeightInitType.XAVIER_NORMAL):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init_type = weight_init_type
        self.weights, self.biases = initialize_weights(self.input_dim, self.output_dim, weight_init_type)
        self.inputs: np.ndarray | None = None
        self.weight_grads: np.ndarray | None = None
        self.bias_grads: np.ndarray | None = None

    def forward(self, inputs) -> np.ndarray:
        # save the inputs for use in the backward pass
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, gradients) -> np.ndarray:
        # Compute gradients of weights and biases
        self.weight_grads = np.dot(self.inputs.T, gradients)
        self.bias_grads = np.sum(gradients, axis=0, keepdims=True)

        # Compute the gradient of the loss with respect to the inputs
        input_gradients = np.dot(gradients, self.weights.T)

        return input_gradients

    def params(self) -> list:
        return [self.weights, self.biases]

    def grads(self) -> list:
        return [self.weight_grads, self.bias_grads]

