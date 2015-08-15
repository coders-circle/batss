import numpy as np
from .dendrite import Dendrite


class Neuron:
    """A neuron with a number of input dendrites.

    Each input dendrite is connected to an input neuron.
    """

    def __init__(self, initial_activation=0, sources=None):
        """Construct a new neuron.

        For each source neuron provided, the constructor creates
        a dendrite to the source with random weight assigned.

        Args:
            initial_activation: The initial activation value for this neuron.
                                Defaults to 0.
            sources: The list of input neurons connected to this neuron.
                     Defaults to None.
        """

        self.activation = initial_activation
        self.dendrites = []
        if sources:
            for source in sources:
                self.dendrites.append(Dendrite(source))


def activate(value):
    """Sigmoid activation function"""

    return 1.0 / (1+np.exp(-value))


def activate_diff(value):
    """Derivative of sigmoid activation function"""

    f = activate(value)
    return f * (1-f)
