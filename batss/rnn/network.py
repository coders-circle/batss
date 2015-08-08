from .neuron import Neuron, activate, activate_diff
from .dendrite import Dendrite
import numpy as np


class Network:
    """A Recurrent Neural Network

    A recurrent neural network that implements the blind signal separation
    training algorithm.
    """

    def __init__(self, num_inputs, num_hidden, num_outputs):
        """Construct a recurrent neural network.

        Args:
            num_inputs: Number of input layer neurons.
            num_hidden: Number of middle/hidden layer neurons.
            num_outputs: Number of output layer neurons.
        """

        # Create the necessary neurons.
        # Combine hidden and output neurons as processor neurons.
        self.inputs = [Neuron() for _ in range(num_inputs)]
        self.processors = [Neuron() for _ in range(num_hidden + num_outputs)]
        self.num_outputs = num_outputs

        # Connect the processor neurons with every neuron.
        for processor in self.processors:
            for inn in self.inputs:
                processor.dendrites.append(Dendrite(inn))
            for pnn in self.processors:
                processor.dendrites.append(Dendrite(pnn))

    def get_outputs(self):
        """Get a list of current output activations from the network.

        Returns:
            list: List of output activation values.
        """

        output_neurons = self.processors[-self.num_outputs:]
        return [neuron.activation for neuron in output_neurons]

    def set_inputs(self, values):
        """Set the activations for the network inputs.

        Args:
            values: List of input activation values.
        """

        for i, neuron in enumerate(self.inputs):
            neuron.activation = values[i]

    def forward(self):
        """Run the recurrent network using current inputs to get outputs"""

        outputs = self._get_y()
        for i, neuron in enumerate(self.processors):
            neuron.activation = outputs[i]

    # We need following matrices.
    # - X = [activations of source neurons of dendrites] as column vector
    # - W = [weights of dendrites] as rectangular matrix
    # - Y = [output activations of weighted sum of input activations]
    #     = f(W*X)

    def _get_x(self):
        # Use dendrites of just one processor neuron
        #  since all other processors are also connected to same neuron sources
        dendrites = self.processors[0].dendrites
        xs = [dendrite.source.activation for dendrite in dendrites]
        return np.matrix(xs).transpose()

    def _get_w(self):
        ws = []
        for pn in self.processors:
            ws.append([dendrite.weight for dendrite in pn.dendrites])

        return np.matrix(ws)

    def _get_y(self):
        # Simply multiply the weight matrix by the activation vector
        # and use activation function in each element
        outputs = (self._get_w() * self._get_x()).tolist()
        outputs = [activate(output[0]) for output in outputs]
        return outputs
