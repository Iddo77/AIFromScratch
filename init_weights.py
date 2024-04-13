from enum import Enum

import numpy as np


class WeightInitType(Enum):
    XAVIER_UNIFORM = 1
    XAVIER_NORMAL = 2
    KAIMING_HE = 3
    RANDOM_UNIFORM = 4
    RANDOM_NORMAL = 5


def initialize_weights(input_dim, output_dim, init_type: WeightInitType):
    if init_type == WeightInitType.XAVIER_UNIFORM:
        # Xavier/Glorot uniform initialization
        # Designed to prevent the vanishing or exploding gradients
        # by initializing weights in a range that is small but not too small, using a uniform distribution.
        limit = np.sqrt(6 / (input_dim + output_dim))
        weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        biases = np.zeros((1, output_dim))
    elif init_type == WeightInitType.XAVIER_NORMAL:
        # Xavier/Glorot normal initialization
        # Similar to the uniform version, this method prevents vanishing or exploding gradients
        # by initializing weights with a standard deviation that is small but not too small, using a normal distribution.
        stddev = np.sqrt(2 / (input_dim + output_dim))
        weights = np.random.normal(0, stddev, (input_dim, output_dim))
        biases = np.zeros((1, output_dim))
    elif init_type == WeightInitType.KAIMING_HE:
        # Kaiming/He initialization
        # Specifically designed for layers with ReLU activation functions to prevent dead neurons
        # by initializing weights with a variance scaled according to the number of input units, using a normal distribution.
        stddev = np.sqrt(2 / input_dim)  # Note: Adjusted for ReLU activations
        weights = np.random.normal(0, stddev, (input_dim, output_dim))
        biases = np.zeros((1, output_dim))
    elif init_type == WeightInitType.RANDOM_UNIFORM:
        # Default uniform initialization
        # Draws samples from a uniform distribution over [0, 1).
        weights = np.random.uniform(size=(input_dim, output_dim))
        biases = np.zeros((1, output_dim))
    elif init_type == WeightInitType.RANDOM_NORMAL:
        # Default normal initialization
        # Draws samples from a standard normal distribution with mean 0 and standard deviation 1.
        weights = np.random.normal(size=(input_dim, output_dim))
        biases = np.zeros((1, output_dim))
    else:
        raise ValueError('Invalid weight initialization type')
    return weights, biases

