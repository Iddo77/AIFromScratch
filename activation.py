import numpy as np

from layer import Layer


class Relu(Layer):
    def __init__(self):
        self.relus: np.ndarray | None = None

    def forward(self, inputs):
        self.relus = np.maximum(0, inputs)
        return self.relus

    def backward(self, gradients):
        """
        Returns the derivative of the relu function with respect to the input, multiplied by the gradients.
        """
        return (self.relus > 0).astype(float) * gradients


class Sigmoid(Layer):
    def __init__(self):
        self.sigmoids: np.ndarray | None = None

    def forward(self, inputs):
        """
        Why the "1 +" in the Denominator?
        The "1 +" in the denominator of the sigmoid function ensures that the output of the function is always between 0 and 1.
        Here's how:
        - The exponential function e^-x ranges from 0 to infinity as x moves from infinity to negative infinity.
        - By adding 1 to e^-x, the range shifts to 1 to infinity,
        ensuring the denominator never becomes zero (which would make the function undefined) and always stays positive.
        - Inverting this with 1/ (the numerator) maps large positive inputs to values close to 1,
         and large negative inputs to values close to 0, with the function smoothly transitioning between these two extremes.
         This behavior is critical for the sigmoid function's role in "squashing" inputs into a fixed range.
        """
        self.sigmoids = 1.0 / (1.0 + np.exp(-inputs))
        return self.sigmoids

    def backward(self, gradients):
        """
        Returns the derivative of the sigmoid function with respect to the input, multiplied by the gradients.
        """
        return self.sigmoids * (1.0 - self.sigmoids) * gradients


class Tanh(Layer):
    def __init__(self):
        self.outputs = None  # To store the output for use in the backward pass

    def forward(self, inputs):
        """
        Apply the hyperbolic tangent function to the inputs using the exponential function,
        with handling for extreme input values to maintain numerical stability.
        """
        # Handle extremes to prevent overflow/underflow
        inputs_clipped = np.clip(inputs, -100, 100)

        # Calculate e^x and e^-x using clipped inputs
        exp_pos = np.exp(inputs_clipped)
        exp_neg = np.exp(-inputs_clipped)

        # Calculate Tanh using the definition (e^x - e^-x) / (e^x + e^-x)
        self.outputs = (exp_pos - exp_neg) / (exp_pos + exp_neg)
        return self.outputs

    def backward(self, gradients):
        """
        Compute the gradient of the loss with respect to the input of this layer,
        using the derivative of the Tanh function. The derivative of Tanh is:
            1 - (tanh(x))^2
        This derivative is then multiplied by the gradient of the loss with respect to the output.
        """
        tanh_derivative = 1 - np.power(self.outputs, 2)
        return tanh_derivative * gradients
