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

        return pickle.load(open(filename, "rb"))

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

        outputs = self._get_y()
        for i, neuron in enumerate(self.processors):
            neuron.activation = activate(outputs[i])

        # store the output samples for future reference
        output_neurons = self.processors[-self.num_outputs:]
        if addsamples:
            self.samples.append(self.get_outputs())

    def train(self):
        """Train the recurrent neural network for bss."""

        # TODO: Implement GPU version of this function.

        # Calculate the delta weight using formula given in the paper.
        cfactor = self._get_cfactor()
        error = self._get_error_vector()

        xm = self._get_x()
        dwtm = cfactor * error * (1.0/(xm.transpose()*xm).tolist()[0][0])
        dwtm = dwtm * xm.transpose()

        # Finally adjust weights of the dendrites.
        for i, pn in enumerate(self.processors):
            for j, dendrite in enumerate(pn.dendrites):
                dendrite.weight += dwtm[i, j]

    # We need the following matrices.
    # - X = [input activations] as column vector
    # - W = [weights of dendrites] as rectangular matrix
    # - Y = [weighted sum of inputs] as column vector = W * X

    def _get_x(self):
        # Use dendrites of just one processor neuron
        # since all other processors are also connected to same neuron sources.
        dendrites = self.processors[0].dendrites
        xs = [dendrite.source.activation for dendrite in dendrites]
        return np.matrix(xs).transpose()

    def _get_w(self):
        ws = []
        for pn in self.processors:
            ws.append([dendrite.weight for dendrite in pn.dendrites])
        return np.matrix(ws)

    def _get_y(self):
        # Simply multiply the weight matrix by the input activation vector
        outputs = (self._get_w() * self._get_x()).transpose().tolist()[0]
        return outputs

    # The training algorithm is based on 2-D system theory
    # developed for RNN by Chow and Fang.

    # For the training algorithm, we need to calculate
    # the cross-correlation values between each pair of output signals
    # and the inverse-diagonal matrix of weighted sum of inputs
    # which I will call the C-Factor.

    def _get_error_vector(self, lags=None):

        # For each hidden neuron, the error is zero
        # and for each output neuron, the error is the cross
        # cross correlation between it and next output neuron.

        vector = [0]*len(self.processors)
        for i in range(self.num_outputs):
            j = len(self.processors) - self.num_outputs + i
            inext = (i+1) % self.num_outputs
            vector[j] = self._get_cross_correlation(i, inext)

        vector = np.matrix([vector]).transpose()
        return vector

    def _get_cross_correlation(self, output1, output2, lag=1):
        limit = len(self.samples) - lag
        if limit <= 0:
            return 1

        sumv = 0
        for m in range(0, limit):
            sumv += self.samples[m][output1] * self.samples[m+lag][output2]
        return sumv / limit

    def _get_cfactor(self):

        # The C-Factor is the inverse of the diagonal matrix
        # formed from weighted-sums of inputs as diagonal elements

        weighted_sums = self._get_y()
        vals = [activate_diff(wsum) for wsum in weighted_sums]

        # Create a diagonal matrix from the vals as diagonal
        # and find its inverse.
        # Actually, inverse of diagonal matrix just has its diagonal
        # elements inverted, so directly find that.
        matrix = np.zeros((len(vals), len(vals))).tolist()
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i == j:
                    matrix[i][i] = 0 if vals[i] == 0 else 1/vals[i]

        # Return the inverse of the diagonal matrix
        return np.matrix(matrix)
