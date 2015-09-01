import numpy as np
import random
from .helpers import activate, activate_diff, get_random_weight


class Layer:
    def __init__(self, num_neurons, prev_layer=None):
        """Constructs a layer with given number of neurons.

        Args:
            num_neurons: Number of neurons in this layer.
            prev_layer: Previous layer which acts as input to this
                        layer. None for input layer.
        """

        # x : Activation vector of the neurons.
        # nets : Vector of weighted sum of inputs of the neurons.
        # deltas : Delta error vector, used to adjust the weights.
        self.x = np.array([0] * num_neurons)
        self.nets = np.array([0] * num_neurons)
        self.deltas = np.array([0] * num_neurons)

        self.prev_layer = prev_layer

        # If previous layer exists, create a weight matrix
        # with random values.
        if prev_layer:
            self.weights = []
            for i in range(num_neurons):

                # Each neuron is connected to all neurons of previous layer
                # plus a constant input of '1' (the weight of which is
                # bias). So total number of weights = num_inputs + 1.

                prev_x_len = len(prev_layer.x) + 1
                w = [get_random_weight(prev_x_len) for _ in range(prev_x_len)]
                self.weights.append(w)

            self.weights = np.matrix(self.weights)

    def forward(self):
        """Use inputs from previous layer to activate neurons of this layer."""

        # Inputs = Previous layer activation vector + [1]
        prev_x = np.append(self.prev_layer.x, 1)

        # Net vector = Weight_Matrix * Inputs
        y = np.dot(self.weights, prev_x)
        self.nets = np.array(y.tolist()).flatten()

        # Activation Vector = Activate(Net Vector)
        self.x = activate(self.nets)

    def adjust_weights(self, rate):
        """Use the delta vector to adjust weights of this layer.

        rate: Learning rate to use during adjustion.
        """

        # Inputs = Previous layer activation vector + [1]
        px = np.matrix(np.append(self.prev_layer.x, 1))

        # Delta-Weight = rate * Delta_Vector.Transpose() * Inputs
        self.weights += rate * np.matrix(self.deltas).transpose() * px
