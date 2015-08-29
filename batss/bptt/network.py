import numpy as np
from random import random
from .helpers import activate, activate_diff
import pickle


class Network:

    def __init__(self, num_inputs, num_hiddens, num_outputs):

        # The neurons' initial activations.
        self.inputs = [0 for _ in range(num_inputs)]
        self.hiddens = [0 for _ in range(num_hiddens)]
        self.outputs = [0 for _ in range(num_outputs)]

        # The weight matrices: input, recurrent, back and output.
        self.win = []
        self.wrec = []
        self.wback = []
        for _ in self.hiddens:
            self.win.append([(random()*2-1)/num_hiddens for _ in range(num_inputs)])
            self.wrec.append([(random()*2-1)/num_hiddens for _ in range(num_hiddens)])
            self.wback.append([(random()*2-1)/num_hiddens for _ in range(num_outputs)])

        self.wout = []
        for _ in self.outputs:
            self.wout.append([(random()*2-1)/num_hiddens for _ in range(num_hiddens)])

        # Convert to numpy matrices.
        self.inputs = np.matrix(self.inputs).transpose()
        self.hiddens = np.matrix(self.hiddens).transpose()
        self.outputs = np.matrix(self.outputs).transpose()
        self.win = np.matrix(self.win)
        self.wrec = np.matrix(self.wrec)
        self.wout = np.matrix(self.wout)
        self.wback = np.matrix(self.wback)

        # The collected samples.
        self.samples = []
        self.hsamples = []
        self.hpotentials = []
        self.opotentials = []

    def load(filename):
        """Load a recurrent neural network from a file.

        Args:
            filename: File to load rnn from.

        Returns:
            Network: A new recurrent neural network loaded from the file.
        """

        newrnn = pickle.load(open(filename, "rb"))
        newrnn.samples = []
        newrnn.hsamples = []
        newrnn.hpotentials = []
        newrnn.opotentials = []
        return newrnn

    def save(self, filename):
        """Save the current neural network to a file.

        Args:
            filename: File to save rnn to.
        """

        pickle.dump(self, open(filename, "wb"))

    def set_inputs(self, inputs):
        self.inputs = np.matrix(inputs).transpose()

    def get_outputs(self):
        return self.outputs.transpose().tolist()[0]

    def forward(self):

        # For hidden neurons.
        vals = self.win * self.inputs
        vals += self.wrec * self.hiddens
        vals += self.wback * self.outputs
        vals = vals.transpose().tolist()[0]
        self.hpotentials.append(vals)
        vals = [activate(v) for v in vals]
        self.hsamples.append(vals)
        self.hiddens = np.matrix(vals).transpose()

        # For output neurons.
        vals = (self.wout * self.hiddens).transpose().tolist()[0]
        self.opotentials.append(vals)
        vals = [activate(v) for v in vals]
        self.samples.append(vals)
        self.outputs = np.matrix(vals).transpose()

    def train(self, input_series, output_series, rate=0.5):

        # Step 1: Forward pass.
        self.samples = []
        self.hsamples = []
        self.hpotentials = []
        self.opotentials = []
        for inputs in input_series:
            self.set_inputs(inputs)
            self.forward()

        hps = []
        for h in self.hpotentials:
            hps.append(np.array([activate_diff(v) for v in h]))
        ops = []
        for o in self.opotentials:
            ops.append(np.array([activate_diff(v) for v in o]))

        # Step 2: Calculate error propagations.
        T = len(input_series)
        dj = [None] * T
        di = [None] * T
        for i in range(T-1, -1, -1):
            err = np.array(output_series[i]) - np.array(self.samples[i])
            if i != T-1:
                err += np.array((di[i+1] * self.wback).tolist()[0])
            dj[i] = err * ops[i]

            err = np.array((dj[i] * self.wout).tolist()[0])
            if i != T-1:
                err += np.array((di[i+1] * self.wrec).tolist()[0])
            di[i] = err * hps[i]

        dj = np.matrix(dj)
        di = np.matrix(di)

        # Step 3: Adjust the weights.
        x = np.roll(np.matrix(self.hsamples), 1, axis=0)
        x[0] = np.array([0] * len(x[0]))
        y = np.roll(np.matrix(self.samples), 1, axis=0)
        y[0] = np.array([0] * len(y[0]))

        self.wrec += rate * di.transpose() * x
        self.win += rate * di.transpose() * np.matrix(input_series)
        self.wout += rate * dj.transpose() * np.matrix(self.hsamples)
        self.wback += rate * di.transpose() * y
