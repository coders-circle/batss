import random
import pickle
import numpy as np

from .helpers import activate, activate_diff
from .layer import Layer


class Network:
    def __init__(self, ni, nh, no):
        """Constructs a neural network with given number of neurons.

        Args:
            ni: Number of input layer neurons.
            nh: Number of hidden layer neurons.
            no: Number of output layer neurons.
        """

        self.input_layer = Layer(ni, None)
        self.hidden_layer = Layer(nh, self.input_layer)
        self.output_layer = Layer(no, self.hidden_layer)

    def save(self, file):
        """Save the neural network data in the given file object.

        Args:
            file: An opened file object for writing in bytes.
        """

        pickle.dump(self.hidden_layer.weights, file)
        pickle.dump(self.output_layer.weights, file)

    def load(self, file):
        """Load the neural network data from the given file object.

        Args:
            file: An opened file object for reading in bytes.
        """

        self.hidden_layer.weights = pickle.load(file)
        self.output_layer.weights = pickle.load(file)

    def set_inputs(self, inputs):
        """Set the activations for the input layer neurons.

        Args:
            inputs: List of input values for each neuron.
        """

        inputs = list(inputs)
        num_inputs = len(self.input_layer.x)
        if num_inputs > len(inputs):
            extra = num_inputs - len(inputs)
            inputs = inputs + [0]*extra

        self.input_layer.x = np.array(inputs[:num_inputs])

    def get_outputs(self):
        """Get the output layer neurons' activations.

        Returns:
            list: Activation values from output layer neurons.
        """

        return list(self.output_layer.x)

    def forward(self):
        """Feed forward the input values through the neural network."""

        self.hidden_layer.forward()
        self.output_layer.forward()

    def train(self, inputs, targets, rate=0.05):
        """Train the neural network with given input and outputs once.

        The backpropagation learning algorithm is implemented for the
        training.

        Args:
            inputs: List of inputs.
            targets: List of target outputs.
            rate: Learning rate for the training.

        Returns:
            Average error of output layer.
        """

        # Step 1: Set the inputs and feed-forward once.
        self.set_inputs(inputs)
        self.forward()

        targets = list(targets)
        num_outputs = len(self.output_layer.x)
        if num_outputs > len(targets):
            extra = num_outputs - len(targets)
            targets = targets + [0]*extra
        targets = np.array(targets[:num_outputs])

        # Step 2: Calculate delta error vectors.

        # For output layer, the delta vector is simply
        # the difference of target and obtained output vectors.
        self.output_layer.deltas = targets - self.output_layer.x

        # For hidden layer, the errors are backpropagated from
        # output layer.

        # dh = W.Transpose() * D
        # where W and D are weight matrix and delta vector of
        # output layer.
        weightsT = self.output_layer.weights.transpose()
        dh = np.dot(weightsT, self.output_layer.deltas)
        # Remove last element that corresponds to bias.
        dh = dh.tolist()[0][:-1]

        # fdash = Differential_Activate(Hidden_Layer.Net)
        # where Net = Weight * Inputs
        fdash = activate_diff(self.hidden_layer.nets).tolist()

        # Finally, delta vector of hidden layer:
        # delta = dh .* fdash
        # .* is component wise multiplication.
        dh = np.multiply(dh, fdash)
        self.hidden_layer.deltas = dh

        # Step 3: Use the delta vectors to adjust the weights.
        self.hidden_layer.adjust_weights(rate)
        self.output_layer.adjust_weights(rate)

        return np.average(self.output_layer.deltas)
