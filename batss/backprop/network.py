import random
import numpy as np
from .helpers import activate, activate_diff
from .layer import Layer


class Network:
    def __init__(self, ni, nh, no):
        self.input_layer = Layer(ni, None)
        self.hidden_layer = Layer(nh, self.input_layer)
        self.output_layer = Layer(no, self.hidden_layer)

    def set_inputs(self, inputs):
        assert(len(self.input_layer.x) <= len(inputs))
        self.input_layer.x = list(inputs)

    def get_outputs(self):
        return self.output_layer.x

    def forward(self):
        self.hidden_layer.forward()
        self.output_layer.forward()

    def train(self, inputs, targets, rate=0.05):
        self.set_inputs(inputs)
        self.forward()

        assert(len(self.output_layer.x) <= len(targets))

        errors = np.array(targets) - np.array(self.output_layer.x)
        self.output_layer.deltas = errors

        weightsT = self.output_layer.weights.transpose()
        dh = np.dot(weightsT, self.output_layer.deltas)
        dh = dh.tolist()[0][:-1]
        fdash = [activate_diff(i) for i in self.hidden_layer.nets]
        dh = np.multiply(dh, fdash)
        self.hidden_layer.deltas = dh

        self.hidden_layer.adjust_weights(rate)
        self.output_layer.adjust_weights(rate)
        return np.average(errors)
