import numpy as np


class Sound:
    def __init__(self, sample, rate):
        self.sample = sample
        self.rate = rate

    def normalize(self):
        self.sample = self.sample/2.**15
        return self

    def denormalize(self):
        self.sample = self.sample*2.**15
        return self

    def reshape(self):
        # Shift using the mean of samples
        # and scale them so that there are minimum zeros
        # after decimal.

        mean = np.mean(self.sample)
        ss = [s-mean for s in self.sample]

        maxval = np.amax(ss)
        factor = 10 ** (int(-np.log10(maxval)))
        self.sample = np.array([s*factor for s in ss])
