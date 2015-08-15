import random


class Dendrite:
    """A dendrite connects a neuron to an input neuron.

    Each neuron has a list of dendrites through which it gets inputs.
    Each dendrite has a weight and a source neuron.
    """

    def __init__(self, source, weight=None):
        """Constructs a dendrite from a given input neuron, with given weight.

        If no weight is provided, a random value is used.

        Agrs:
            source: The input neuron.
            weight: The weight for this dendrite. Defaults to None.
        """

        if weight:
            self.weight = weight
        else:
            self.weight = random.random()
        self.source = source
