from rnn import network
from sound import sound, sound_manager
import numpy as np


def create_rnn(filename, inputs, hiddens, outputs):
    """Create a new recurrent neural network in a file.

    Args:
        filename: File to store the rnn in.
        inputs: Number of input neurons.
        hiddens: Number of hidden neurons.
        outputs: Number of output neurons.
    """

    print("Creating new neural network")
    rnn = network.Network(inputs, hiddens, outputs)
    rnn.save(filename)
    print("Saved new neural network to file: ", filename)


def train_rnn(filename, input_files, output_files, learning_rate=0.05,
             iterations=1, num_samples=2000):
    """Train a recurrent neural network with given sounds.

    Args:
        filename: File to load rnn from.
        input: List of input sound files for training.
                     Every channel of every sound file is used.
        output: List of output sound files for training.
                      Only one channel of each sound file is used.
        learning_rate: Learning rate of the network.
        iterations: Number of iterations this training repeats for this set.
        num_samples: Number of first 'n' samples of inputs and outputs
                     to use during training.
    """

    print("Loading neural network from file: ", filename)
    rnn = network.Network.load(filename)
    smanager = sound_manager.SoundManager()

    print("Number of traning samples: ", num_samples)
    print("Collecting samples...")

    # Get every channel from every input sound file.
    inputs = []
    for i in input_files:
        sounds = smanager.read_file(i)
        if type(sounds) == tuple:
            for s in sounds:
                inputs.append(list(s.sample)[0:num_samples])
        else:
            inputs.append(list(sounds.sample)[0:num_samples])

    # Get one channel from every output sound file.
    outputs = []
    for o in output_files:
        sounds = smanager.read_file(o)
        if type(sounds) == tuple:
            outputs.append(list(sounds[0].sample)[0:num_samples])
        else:
            outputs.append(list(sounds.sample)[0:num_samples])

    if len(rnn.inputs) > len(inputs):
        raise Exception("Not enough inputs provided for training")
    if rnn.num_outputs > len(outputs):
        raise Exception("Not enough outputs provided for training")

    input_series = []
    for j in range(num_samples):
        samples = [i[j] for i in inputs]
        input_series.append(samples)

    output_series = []
    for j in range(num_samples):
        samples = [o[j] for o in outputs]
        output_series.append(samples)

    print("Training...")
    for i in range(iterations):
        print("Iteration #", i)
        rnn.train(input_series, learning_rate, output_series)

    print("Done")
    rnn.save(filename)
    print("Saved trained network to file: ", filename)


def separate(filename, input_files, output_files):
    """Use recurrent neural network to separate given input sounds.

    Args:
        filename: File to load neural network from.
        input_files: Input sound files that need to be separated.
        output_files: Files to output separated sounds.
    """

    print("Loading neural network from file: ", filename)
    rnn = network.Network.load(filename)
    smanager = sound_manager.SoundManager()

    # Get every channel from every input sound file.
    print("Collecting input samples...")
    inputs = []
    inputsounds = []
    rate = 0
    for i in input_files:
        sounds = smanager.read_file(i)
        if type(sounds) == tuple:
            for s in sounds:
                inputs.append(list(s.sample))
                rate = s.rate
                inputsounds.append(s)
        else:
            inputs.append(list(sounds.sample))
            inputsounds.append(sounds)
            rate = sounds.rate

    print("Separating...")
    for i in range(len(inputs[0])):
        rnn.set_inputs([inp[i] for inp in inputs])
        rnn.forward()

    print("Saving output files...")
    outputs = []
    for i, o in enumerate(output_files):
        output = sound.Sound(np.asarray([s[i] for s in rnn.samples]), rate)
        outputs.append(output)
        smanager.save_file(output, o)

    print("Done")
    smanager.plot([tuple(inputsounds), tuple(outputs)])


create_rnn("test.rnn", 2, 5, 2)

train_rnn("test.rnn",
         input_files=["sound/WAV/X_rss.wav"],
         output_files=["sound/WAV/Y1_rss.wav", "sound/WAV/Y2_rss.wav"],
         learning_rate=0.05, iterations=10, num_samples=1000)

separate("test.rnn",
         input_files=["sound/WAV/X_rss.wav"],
         output_files=["sound/WAV/output1.wav", "sound/WAV/output2.wav"])
