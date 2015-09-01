import random
import numpy as np


def activate(value):
    """Bipolar Sigmoid activation function"""

    t = 2.0 / (1+np.exp(-value)) - 1
    return t


def activate_diff(value):
    """Derivative of sigmoid activation function"""

    f = activate(value)
    return 0.5 * (1+f) * (1-f)


def get_random_weight(div_factor=1):
    """Get a random value in range (-.5, .5) divided by given factor"""

    return (random.random() - 0.5) / (div_factor)
