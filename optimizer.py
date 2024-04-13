import numpy as np

from layer import Layer


class Optimizer:
    """
    An optimizer updates the weights of a layer (in place) using information known by the layer or the optimizer.
    """
    def step(self, layer: Layer):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, layer: Layer):
        parameters = layer.params()
        gradients = layer.grads()

        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            param -= self.learning_rate * grad


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}  # To store the velocity vector for each parameter

    def step(self, layer: Layer):
        parameters = layer.params()
        gradients = layer.grads()

        # Initialize velocity for all parameters if not already done
        if not self.velocity:
            for param in parameters:
                self.velocity[id(param)] = np.zeros_like(param)

        # Update each parameter
        for i, (param, grad) in enumerate(zip(parameters, gradients)):

            current_velocity = self.velocity[id(param)]

            # Update the velocity with a blend of momentum and the current gradient.
            # This blend allows the optimizer to "remember" some of the previous direction
            # it was moving in (scaled by `momentum`), while also adjusting based on the
            # new information from the current gradient (scaled by `1 - momentum`).
            # This helps in smoothing out the updates and provides faster convergence,
            # especially in landscapes with ravines and plateaus.
            updated_velocity = self.momentum * current_velocity + (1 - self.momentum) * grad
            self.velocity[id(param)] = updated_velocity

            #  Update the parameter by applying the learning rate to the velocity
            param -= self.learning_rate * updated_velocity
