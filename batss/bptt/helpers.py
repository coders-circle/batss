import numpy as np


def activate(value):
    """Bipolar Sigmoid activation function"""

    return 2.0 / (1+np.exp(-value)) - 1


def activate_diff(value):
    """Derivative of sigmoid activation function"""

    f = activate(value)
    return 0.5 * (1+f) * (1-f)
