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
