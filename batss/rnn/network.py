from neuron import Neuron
from dendrite import Dendrite


class Network:
    """A Recurrent Neural Network

    A recurrent neural network that implements the blind signal separation
    training algorithm.
    """

    def __init__(self, num_inputs, num_hidden, num_outputs):
        """Construct a recurrent neural network.

        Args:
            num_inputs: number of input layer neurons
            num_hidden: number of middle/hidden layer neurons
            num_outputs: number of output layer neurons
        """

        # Create the necessary neurons
        self.inputs = [Neuron() for _ in range(num_inputs)]
        self.hiddens = [Neuron() for _ in range(num_hidden)]
        self.ouputs = [Neuron() for _ in range(num_outputs)]
