import shutil
import pickle
import os.path
import time
import numpy as np

from backprop import network
from sound import sound, sound_manager


def ordinal(n):
    s = "tsnrhtdd"
    return "%d%s" % (n, s[(int(n/10) % 10 != 1) * (n % 10 < 4)*n % 10::4])


def new_network(num_samples, num_inputs, num_outputs, num_hidden_layers=1):

    # Calculate required number of neurons.
    inputs = int(num_samples * num_inputs)
    outputs = int(num_samples * num_outputs)
    nhiddens = int(np.mean((inputs, outputs)))
    hiddens = [nhiddens for _ in range(num_hidden_layers)]

    return network.Network(inputs, hiddens, outputs)


def save_network(filename, num_samples, nn):
    num_inputs = len(nn.input_layer.x) / num_samples
    num_outputs = len(nn.output_layer.x) / num_samples
    num_hidden_layers = len(nn.hidden_layers)
    with open(filename, "wb") as file:
        pickle.dump(num_samples, file)
        pickle.dump(num_inputs, file)
        pickle.dump(num_outputs, file)
        pickle.dump(num_hidden_layers, file)
        nn.save(file)


def load_network(filename):
    with open(filename, "rb") as file:
        samples = pickle.load(file)
        ni = pickle.load(file)
        no = pickle.load(file)
        nhl = pickle.load(file)
        nn = new_network(samples, ni, no, nhl)
        nn.load(file)

    return nn, samples


def create_network(filename, num_samples, num_inputs=1, num_outputs=1,
                   num_hidden_layers=1):
    """Create a new neural network in a file.

    The number of input, hidden and output neurons are automatically
    calculated according to the given arguments.

    Args:
        filename: File to store the network data in.
        num_samples: Number of samples processed once by the network.
        num_inputs: Number of input signals to the network.
        num_outputs: Number of output signals generated by the network.
        num_hidden_layers: Number of hidden layers used in the network.
    """

    print("Creating new neural network")
    nn = new_network(num_samples, num_inputs, num_outputs, num_hidden_layers)

    # Make a backup, in case such file already exists.
    if os.path.isfile(filename):
        print("File already exists.")
        print("Creating backup of old file:", filename+"~")
        shutil.copyfile(filename, filename+"~")

    save_network(filename, num_samples, nn)

    print("Saved new neural network to file:", filename)


def train_network(filename, input_files, output_files, learning_rate=0.03,
                  iterations=30, offset=0, frames=None):
    """Train a neural network with given sound files.

    Args:
        filename: File to load neural network from.
        input: List of input sound files for training.
               Every channel of every sound file is used.
        output: List of output sound files for training.
                Only one channel of each sound file is used.
        learning_rate: Learning rate of the network.
        iterations: Number of iterations this training repeats for this set.
        offset: Offset of sound samples from beginning to use for training.
        frames: Number of frames to use for training.
    """

    print("Loading neural network from file: ", filename)
    nn, num_samples = load_network(filename)
    smanager = sound_manager.SoundManager()
    num_inputs = int(len(nn.input_layer.x) / num_samples)
    num_outputs = int(len(nn.output_layer.x) / num_samples)

    # Display some info about the neural network.
    nhl = len(nn.hidden_layers)
    print("Number of neurons:")
    print("Input Layer: ", len(nn.input_layer.x))
    if nhl > 0:
        print("Hidden Layer: ", len(nn.hidden_layers[0].x))
    print("Output Layer: ", len(nn.output_layer.x))
    print("Number of hidden layers:", nhl)
    print("Number of samples per frame:", num_samples)
    print("Number of iterations:", iterations)
    print("Learning rate:", learning_rate)
    print("Collecting samples...")

    # Log-file name.
    log_file = os.path.splitext(filename)[0] + "_"
    log_file += time.strftime("%b-%d-%Y_%H%M%S")
    log_file += ".log"

    # Get every channel from every input sound file.
    inputs = []
    for i in input_files:
        if len(inputs) == num_inputs:
            break
        sounds = smanager.read_file(i)
        if type(sounds) is tuple:
            for s in sounds:
                if len(inputs) == num_inputs:
                    break
                inputs.append(list(s.sample))
        else:
            inputs.append(list(sounds.sample))

    # Get one channel from every output sound file.
    outputs = []
    for o in output_files:
        if len(outputs) == num_outputs:
            break
        sounds = smanager.read_file(o)
        if type(sounds) is tuple:
            outputs.append(list(sounds[0].sample))
        else:
            outputs.append(list(sounds.sample))

    if num_inputs > len(inputs):
        raise Exception("Not enough inputs provided for training")
    if num_outputs > len(outputs):
        raise Exception("Not enough outputs provided for training")


    if not frames:
        frames = int(len(outputs[0])/num_samples)
    print("Number of frames to train:", frames)

    f = open(log_file, "w")

    for k in range(iterations):
        oi = ordinal(k+1)
        print(oi, "iteration")

        errs = []
        for j in range(frames):
            off = offset + j * num_samples
            input_series = [l[off:off+num_samples] for l in inputs]
            input_series = sum(input_series, [])
            output_series = [l[off:off+num_samples] for l in outputs]
            output_series = sum(output_series, [])
            e = nn.train(input_series, output_series, learning_rate)
            errs.append(e)

        aerr = np.average(errs)
        if log_file:
            f.write("Iteration: " + str(k) + " Average Error: " + str(aerr))
            f.write("\n")

    f.close()

    print("Done")
    # Make a backup, in case this training has corrupted the original data.
    if os.path.isfile(filename):
        shutil.copyfile(filename, filename+"~")
    save_network(filename, num_samples, nn)
    print("Saved trained network to file: ", filename)


def separate(filename, input_files, output_files, extra=None):
    """Use recurrent neural network to separate given input sounds.

    Args:
        filename: File to load neural network from.
        input_files: Input sound files that need to be separated.
        output_files: Files to output separated sounds.
        extra: Extra sound files to plot.
    """

    print("Loading neural network from file: ", filename)
    nn, num_samples = load_network(filename)
    smanager = sound_manager.SoundManager()
    num_inputs = int(len(nn.input_layer.x) / num_samples)

    # Get every channel from every input sound file.
    print("Collecting input samples...")
    inputs = []
    inputsounds = []
    for i in input_files:
        if len(inputs) == num_inputs:
            break
        sounds = smanager.read_file(i)
        if type(sounds) is tuple:
            for s in sounds:
                if len(inputs) == num_inputs:
                    break
                inputs.append(list(s.sample))
                inputsounds.append(s)
        else:
            inputs.append(list(sounds.sample))
            inputsounds.append(sounds)

    print("Separating...")
    outputs = []
    for i in range(int(len(inputs[0])/num_samples)):
        off = i * num_samples
        input_series = [l[off:off+num_samples] for l in inputs]
        input_series = sum(input_series, [])
        nn.set_inputs(input_series)
        nn.forward()
        outputs.append(nn.get_outputs())

    print("Saving output files...")
    outputsounds = []
    for o in range(int(len(nn.output_layer.x)/num_samples)):
        off = o * num_samples
        output_series = [l[off:off+num_samples] for l in outputs]
        output_series = sum(output_series, [])
        rate = inputsounds[0].rate
        output = sound.Sound(np.array(output_series), rate)
        outputsounds.append(output)
        smanager.save_file(output, output_files[o])

    print("Done")
    titles = ["Input", "Output", "Extra"]
    if extra:
        es = []
        for ex in extra:
            es.append(smanager.read_file(ex)[0])
        smanager.plot([tuple(inputsounds), tuple(outputsounds), tuple(es)], titles)
    else:
        smanager.plot([tuple(inputsounds), tuple(outputsounds)], titles)


if __name__ == "__main__":
    sm = sound_manager.SoundManager()

    filename = "sample_networks/2_speeches.nn"
    create_network(filename, 1000, 2, 2, 1)

    train_network(filename,
                  input_files=["sound/WAV/X_rss.wav"],
                  output_files=["sound/WAV/Y1_rss.wav", "sound/WAV/Y2_rss.wav"],
                  learning_rate=0.03, iterations=20,
                  offset=0, frames=None)

    extra1 = "sound/WAV/Y1_rss.wav"
    extra2 = "sound/WAV/Y2_rss.wav"
    separate(filename,
             input_files=["sound/WAV/X_rss.wav"],
             output_files=["sound/WAV/output1.wav", "sound/WAV/output2.wav"],
             extra=[extra1, extra2])

    # filename = "sample_networks/test1.nn"
    # create_network(filename, 1000, 1, 1, 1)

    # train_network(filename,
    #               input_files=["sound/WAV/X_rsm2.wav"],
    #               output_files=["sound/WAV/Y1_rsm2.wav"],
    #               learning_rate=0.03, iterations=20,
    #               offset=0, frames=None)

    # separate(filename,
    #          input_files=["sound/WAV/X_rsm2.wav"],
    #          output_files=["sound/WAV/output.wav"],
    #          extra=["sound/WAV/Y1_rsm2.wav"])
