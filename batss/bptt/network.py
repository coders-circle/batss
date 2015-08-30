import numpy as np
import pyopencl as cl
from random import random
from .helpers import activate, activate_diff
from clw import clwrapper, kernel
import pickle
import copy

CL_INITIALIZED = False


def init_cl():
    if CL_INITIALIZED:
        return

    global clw, mf, matmult, matadd
    clw = clwrapper.CL()
    mf = cl.mem_flags
    matmult = kernel.Kernel(clw)
    matadd = kernel.Kernel(clw)

    matmult.load('matmult.cl')
    matadd.load('matadd.cl')


class Network:

    def __init__(self, num_inputs, num_hiddens, num_outputs, use_gpu=False):
        """Create a new recurrent neural network.

        Create a new recurrent neural network that implements
        backpropagation through time as its training algorithm.

        Args:
            num_inputs: Number of input layer neurons.
            num_hiddens: Number of hidden layer neurons.
            num_outputs: Number of output layer neurons.
            use_gpu = Whether or not to use GPU for processing.
        """

        # The neurons' initial activations.
        self.inputs = [0 for _ in range(num_inputs)]
        self.hiddens = [0 for _ in range(num_hiddens)]
        self.outputs = [0 for _ in range(num_outputs)]

        # The weight matrices: input, recurrent, back and output.
        self.win = []
        self.wrec = []
        self.wback = []
        for _ in self.hiddens:
            win = [(random()-0.5)/num_hiddens for _ in range(num_inputs)]
            self.win.append(win)
            wrec = [(random()-0.5)/num_hiddens for _ in range(num_hiddens)]
            self.wrec.append(wrec)
            wback = [(random()-0.5)/num_hiddens for _ in range(num_outputs)]
            self.wback.append(wback)

        self.wout = []
        self.wiout = []
        for _ in self.outputs:
            wout = [(random()*2-1)/num_hiddens for _ in range(num_hiddens)]
            self.wout.append(wout)
            wiout = [(random()*2-1)/num_hiddens for _ in range(num_inputs)]
            self.wiout.append(wiout)

        # Convert to numpy matrices.
        self.inputs = np.matrix(self.inputs, dtype=np.float32).transpose()
        self.hiddens = np.matrix(self.hiddens, dtype=np.float32).transpose()
        self.outputs = np.matrix(self.outputs, dtype=np.float32).transpose()
        self.win = np.matrix(self.win, dtype=np.float32)
        self.wrec = np.matrix(self.wrec, dtype=np.float32)
        self.wout = np.matrix(self.wout, dtype=np.float32)
        self.wback = np.matrix(self.wback, dtype=np.float32)
        self.wiout = np.matrix(self.wiout, dtype=np.float32)
        self.use_gpu = use_gpu

        # device variables
        if use_gpu:
            init_cl()
            self.d_inputs = cl.Buffer(clw.get_context(),
                                      mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=self.inputs)
            self.d_hiddens = cl.Buffer(clw.get_context(),
                                       mf.READ_WRITE, size=self.hiddens.nbytes)
            self.d_outputs = cl.Buffer(clw.get_context(),
                                       mf.READ_WRITE, size=self.outputs.nbytes)
            self.d_win = cl.Buffer(clw.get_context(),
                                   mf.READ_WRITE, size=self.win.nbytes)
            self.d_wrec = cl.Buffer(clw.get_context(),
                                    mf.READ_WRITE, size=self.wrec.nbytes)
            self.d_wback = cl.Buffer(clw.get_context(),
                                     mf.READ_WRITE, size=self.wback.nbytes)
            self.d_wiout = cl.Buffer(clw.get_context(),
                                     mf.READ_WRITE, size=self.wiout.nbytes)

        self._reset_samples()

    def _reset_samples(self):
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

        newrnn = None
        with open(filename, "rb") as f:
            ni = pickle.load(f)
            nh = pickle.load(f)
            no = pickle.load(f)
            ug = pickle.load(f)
            newrnn = Network(ni, nh, no, ug)
            newrnn.win = pickle.load(f)
            newrnn.wrec = pickle.load(f)
            newrnn.wout = pickle.load(f)
            newrnn.wback = pickle.load(f)
            newrnn.wiout = pickle.load(f)
        return newrnn

    def save(self, filename):
        """Save the current neural network to a file.

        Args:
            filename: File to save rnn to.
        """

        with open(filename, "wb") as f:
            pickle.dump(len(self.inputs), f)
            pickle.dump(len(self.hiddens), f)
            pickle.dump(len(self.outputs), f)
            pickle.dump(self.use_gpu, f)
            pickle.dump(self.win, f)
            pickle.dump(self.wrec, f)
            pickle.dump(self.wout, f)
            pickle.dump(self.wback, f)
            pickle.dump(self.wiout, f)

    def set_inputs(self, inputs):
        """Set the activations for the input neurons.

        Args:
            inputs: List of input values to the network.
        """

        self.inputs = np.matrix(inputs).transpose()

    def get_outputs(self):
        """Get the activations from the output neurons.

        Args:
            list: List of output values from the network.
        """

        return self.outputs.transpose().tolist()[0]

    def forward(self):
        """Forward the inputs through the network to get outputs."""

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
        vals = self.wout * self.hiddens
        vals += self.wiout * self.inputs
        vals = vals.transpose().tolist()[0]
        self.opotentials.append(vals)
        vals = [activate(v) for v in vals]
        self.samples.append(vals)
        self.outputs = np.matrix(vals).transpose()

    def d_forward(self):
        resultSize = self.win.shape[0]*self.hiddens.shape[1]

        d_vals = cl.Buffer(clw.get_context(),
                           mf.READ_WRITE, size=resultSize)
        d_buff = cl.Buffer(clw.get_context(),
                           mf.READ_WRITE, size=resultSize)

        matmult.execute((self.win.shape[0], self.hiddens.shape[1]), None,
                        d_vals, self.d_win, self.d_inputs,
                        self.d_win.shape[0], self.d_hiddens.shape[0])
        matmult.execute((self.wrec.shape[0], self.hiddens.shape[1]), None,
                        d_buff, self.d_wrec, self.d_hiddens,
                        self.d_wrec.shape[0], self.d_hiddens.shape[0])
        matadd.execute(resultSize, None, d_vals, d_vals, d_buff)

    def train(self, input_series, output_series, rate=0.5):
        """Train the network using BPTT using given supervise-set.

        Args:
            input_series: Sequence of list of inputs at discreet time.
            output_series: Sequence of list of corresponding outputs.
            rate: Learning rate of the neural network.

        Returns:
            Average error during the training.
        """

        # Step 1: Forward pass.
        self._reset_samples()
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
        errs = []
        for i in range(T-1, -1, -1):
            err = np.array(output_series[i]) - np.array(self.samples[i])
            errs.append(err)
            # if i == 2 or i == 10:
            #     print(err, output_series[i], self.samples[i])
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
        self.wiout += rate * dj.transpose() * np.matrix(input_series)
        self.wback += rate * di.transpose() * y

        errs = [np.average(err) for err in errs]
        errs = np.average(errs)
        return errs

    def backup(self):
        self.bwrec = copy.copy(self.wrec)
        self.bwin = copy.copy(self.win)
        self.bwout = copy.copy(self.wout)
        self.bwiout = copy.copy(self.wiout)
        self.bwback = copy.copy(self.wback)

    def restore(self):
        self.wrec = copy.copy(self.bwrec)
        self.win = copy.copy(self.bwin)
        self.wout = copy.copy(self.bwout)
        self.wiout = copy.copy(self.bwiout)
        self.wback = copy.copy(self.bwback)
