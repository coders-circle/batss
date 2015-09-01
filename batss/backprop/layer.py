import numpy as np
import random
from .helpers import activate, activate_diff, get_random_weight


class Layer:
    def __init__(self, num_neurons, prev_layer=None):
        """Constructs a layer with given number of neurons.

        Args:
            num_neurons: Number of neurons in this layer.
            prev_layer: Previous layer which acts as input to this
                        layer.
        """

        self.x = [0] * num_neurons
        self.nets = [0] * num_neurons
        self.deltas = [0] * num_neurons
        self.prev_layer = prev_layer

        if prev_layer:
            self.weights = []
            for i in range(num_neurons):
                prev_x_len = len(prev_layer.x)
                n = num_neurons
                w = [get_random_weight(n) for _ in range(prev_x_len+1)]
                self.weights.append(w)

            self.weights = np.matrix(self.weights)

    def forward(self):
        prev_x = self.prev_layer.x + [1]
        y = np.dot(self.weights, prev_x)
        self.nets = y.tolist()[0]
        self.x = [activate(i) for i in self.nets]

    def adjust_weights(self, rate):
        px = np.matrix(self.prev_layer.x + [1])
        self.weights +=  rate * np.matrix(self.deltas).transpose() * px
