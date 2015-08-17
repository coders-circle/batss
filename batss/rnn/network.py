from .neuron import Neuron, activate, activate_diff
from .dendrite import Dendrite
import numpy as np
import pickle


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

        # Create an empty list to store the output samples
        self.samples = []

    def load(filename):
        """Load a recurrent neural network from a file.

        Args:
            filename: File to load rnn from.

        Returns:
            Network: A new recurrent neural network loaded from the file.
        """

        newrnn = pickle.load(open(filename, "rb"))
        newrnn.samples = []
        return newrnn

    def save(self, filename):
        """Save the current neural network to a file.

        Args:
            filename: File to save rnn to.
        """

        pickle.dump(self, open(filename, "wb"))

    def get_outputs(self):
        """Get the activations from the network outputs.

        Returns:
            List: Activation values of output neurons.
        """

        output_neurons = self.processors[-self.num_outputs:]
        return [n.activation for n in output_neurons]

    def set_inputs(self, values):
        """Set the activations for the network inputs.

        Args:
            values: List of input activation values.
        """

        for i, neuron in enumerate(self.inputs):
            neuron.activation = values[i]

    def forward(self, addsamples=True):
        """Run the recurrent network using current inputs to get outputs."""

        # TODO: Implement GPU version of this function.

        outputs = self._get_z().transpose().tolist()[0]
        for i, neuron in enumerate(self.processors):
            neuron.activation = activate(outputs[i])

        # store the output samples for future reference
        output_neurons = self.processors[-self.num_outputs:]
        self.samples.append(self.get_outputs())

    def train(self, input_series, learning_rate, output_series=None):
        """Train the recurrent neural network for bss.

        The real-time recurrent algorithm is used. While
        the original algorithm was for supervised learning,
        I have changed the error calculations so it may (or may not)
        work for unsupervised learning.

        Args:
            input_series: Series of inputs to train the network
                          i.e. complete input signals in the form
                          [[input1, input2], [input1, input2],...].
            learning_rate: Learning rate of network.
            output_series: Optional. The desired outputs of rnn for
                           supervised training. None for unsupervised.
        """

        neurons = self.inputs + self.processors

        self._p = {}
        for i in range(len(self.processors)):
            for j in range(len(neurons)):
                for k in range(len(self.processors)):
                    self._p[(i, j, k)] = 0.0

        for iinput, inputs in enumerate(input_series):
            self.set_inputs(inputs)
            self.forward()

            if output_series:
                errors = self._get_errors_sup(output_series[iinput])
            else:
                errors = self._get_errors_unsup()

            vals = self._get_z().transpose().tolist()[0]
            fdash = [activate_diff(v) for v in vals]

            newps = {}
            for i, p in enumerate(self.processors):
                for j, dn in enumerate(p.dendrites):
                    sumv = 0
                    for k in range(len(self.processors)):
                        newp = fdash[k] * self._get_p(i, j, k)
                        newps[(i, j, k)] = newp
                        sumv += errors[k] * newp

                    dwt = sumv * learning_rate
                    dn.weight += dwt

            self._p = newps

    # We need the following matrices.
    # - X = [input activations] as column vector
    # - Y = [previous processor activations] as column vector
    # - W = [weights of dendrites] as rectangular matrix
    # - Z = [weighted sum of inputs] as column vector = W * X
    # - errors = [error for each processor]

    def _get_x(self):
        # Use dendrites of just one processor neuron
        # since all other processors are also connected to same neuron sources.
        dendrites = self.processors[0].dendrites
        xs = [dendrite.source.activation for dendrite in dendrites]
        return np.matrix(xs).transpose()

    def _get_y(self):
        return [n.activation for p in self.processors]

    def _get_w(self):
        ws = []
        for pn in self.processors:
            ws.append([dendrite.weight for dendrite in pn.dendrites])
        return np.matrix(ws)

    def _get_z(self):
        # Simply multiply the weight matrix by the input activation vector
        outputs = (self._get_w() * self._get_x())
        return outputs

    # TODO:
    # Error vector for supervised training.
    def _get_errors_sup(self, targets):
        outputs = self.get_outputs()
        errors = [0]*len(self.processors)
        for i in range(self.num_outputs):
            j = len(self.processors) - self.num_outputs + i
            errors[j] = targets[i] - outputs[i]
        return errors

    # Error vector for unsupervised training.
    def _get_errors_unsup(self, lags=None):

        # For each hidden neuron, the error is zero
        # and for each output neuron, the error is the cross
        # cross correlation between it and next output neuron.

        errors = [0]*len(self.processors)
        for i in range(self.num_outputs):
            j = len(self.processors) - self.num_outputs + i
            inext = (i+1) % self.num_outputs
            errors[j] = self._get_cross_correlation(i, inext)
        return errors

    def _get_cross_correlation(self, output1, output2, lag=1):
        limit = len(self.samples) - lag
        if limit <= 0:
            return 1

        sumv = 0
        for m in range(0, limit):
            sumv += self.samples[m][output1] * self.samples[m+lag][output2]
        return sumv / limit

    # Training helpers

    def _get_p(self, i, j, k):

        pnode = self.processors[k]
        sumv = 0

        pdendrites = pnode.dendrites[len(self.inputs):]
        for ii, dn in enumerate(pdendrites):
            sumv += dn.weight * self._p[(i, j, ii)]

        if i == k:
            sumv += (self.inputs + self.processors)[j].activation

        return sumv
