import json

import numpy as np


class Layer:
    def forward(self, inputs) -> np.ndarray:
        """
        Forward pass of the layer.
        :param inputs: input to the layer.
        """
        raise NotImplementedError

    def backward(self, gradients) -> np.ndarray:
        """
        Backward propagation of the layer.
        :param gradients: gradient of the loss with respect to the output.
        """
        raise NotImplementedError

    def params(self) -> list:
        """
        Returns a list of parameters for the layer. Default is empty because not all layers have trainable parameters.
        """
        return []

    def grads(self) -> list:
        """
        Returns a list of gradients for the layer's parameters.
        Default is empty because not all layers have trainable parameters.
        """
        return []

    def save_weights(self, filename: str):
        weights = list(self.params())
        weights = [param.tolist() for param in weights if isinstance(param, np.ndarray)]
        with open(filename, 'w') as f:
            json.dump(weights, f)

    def load_weights(self, filename: str):
        with open(filename) as f:
            weights = json.load(f)

        for param, weight in zip(self.params(), weights):
            if isinstance(param, np.ndarray):
                weight_array = np.array(weight).reshape(param.shape)
                # Update the values of param in place
                param[:] = weight_array
