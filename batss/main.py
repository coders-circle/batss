from bptt import network
from sound import sound, sound_manager
import numpy as np
import shutil
import os.path


def ordinal(n):
    s = "tsnrhtdd"
    return "%d%s" % (n, s[(int(n/10) % 10 != 1) * (n % 10 < 4)*n % 10::4])


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
    # Make a backup, in case such file already exists.
    if os.path.isfile(filename):
        print("File already exists.")
        print("Creating backup of old file:", filename+"~")
        shutil.copyfile(filename, filename+"~")
    rnn.save(filename)
    print("Saved new neural network to file:", filename)


def train_rnn(filename, input_files, output_files, learning_rate=0.3,
              iterations=30, num_samples=10, offset=0, frames=100,
              log_file=None):
    """Train a recurrent neural network with given sounds.

    Args:
        filename: File to load rnn from.
        input: List of input sound files for training.
               Every channel of every sound file is used.
        output: List of output sound files for training.
                Only one channel of each sound file is used.
        learning_rate: Learning rate of the network.
        iterations: Number of iterations this training repeats for this set.
        num_samples: Number of 'n' samples of inputs and outputs
                     to use during training.
        offset: Offset of sound samples from beginning to use for training.
        frames: Number of frames each consisting of 'n' samples.
        log_file: File to save log of error and learning rate per iteration.
    """

    print("Loading neural network from file: ", filename)
    rnn = network.Network.load(filename)
    smanager = sound_manager.SoundManager()

    # Display some info about the neural network.
    print("Number of neurons:")
    print("Input Layer: ", len(rnn.inputs))
    print("Hidden Layer: ", len(rnn.hiddens))
    print("Output Layer: ", len(rnn.outputs))
    print("Number of frames to train: ", frames)
    print("Number of samples per frame: ", num_samples)
    print("Number of iterations: ", iterations)
    print("Collecting samples...")

    # Get every channel from every input sound file.
    input_frames = []
    input_sounds = {}
    for k in range(frames):
        inputs = []
        off = offset + num_samples*k
        nindex = off + num_samples
        for i in input_files:
            if len(inputs) == len(rnn.inputs):
                break
            if i not in input_sounds:
                print("Reading ", i)
                sounds = smanager.read_file(i)
                input_sounds[i] = sounds
            else:
                sounds = input_sounds[i]
            if type(sounds) == tuple:
                for s in sounds:
                    if len(inputs) == len(rnn.inputs):
                        break
                    inputs.append(list(s.sample)[off:nindex])
            else:
                inputs.append(list(sounds.sample)[off:nindex])
        input_frames.append(inputs)

    # Get one channel from every output sound file.
    output_frames = []
    output_sounds = {}
    for k in range(frames):
        outputs = []
        off = offset + num_samples*k
        nindex = off + num_samples
        for o in output_files:
            if o not in output_sounds:
                print("Reading ", o)
                sounds = smanager.read_file(o)
                output_sounds[o] = sounds
            else:
                sounds = output_sounds[o]
            if type(sounds) == tuple:
                outputs.append(list(sounds[0].sample)[off:nindex])
            else:
                outputs.append(list(sounds.sample)[off:nindex])
        output_frames.append(outputs)

    if len(rnn.inputs) > len(inputs):
        raise Exception("Not enough inputs provided for training")
    if len(rnn.outputs) > len(outputs):
        raise Exception("Not enough outputs provided for training")

    if log_file:
        f = open(log_file, "w")

    print("Training...")
    last_err = 999999
    for iteration in range(iterations):
        oi = ordinal(iteration+1)
        print(oi, "iteration", "Learning Rate", learning_rate)

        errs = []
        rnn.backup()
        for k in range(frames):
            input_series = []
            for j in range(num_samples):
                samples = [i[j] for i in input_frames[k]]
                input_series.append(samples)

            output_series = []
            for j in range(num_samples):
                samples = [o[j] for o in output_frames[k]]
                output_series.append(samples)

            # rate = learning_rate
            # if annealing_time > 0:
            #     rate /= 1 + iteration/annealing_time

            e = rnn.train(input_series, output_series, learning_rate)
            errs.append(e)

        aerr = np.average(errs)

        if log_file:
            output = "Average error: " + str(aerr) + \
                     " Learning Rate: " + str(learning_rate) + "\n"
            f.write(output)

        diff_err = abs(aerr) - abs(last_err)
        if diff_err <= 0:
            learning_rate += 5/100 * learning_rate
        elif diff_err > 10e-10:
            rnn.restore()
            learning_rate *= 1/2
        else:
            learning_rate -= 5/100 * learning_rate
        last_err = aerr

    if log_file:
        f.close()

    print("Done")
    # Make a backup, in case this training has corrupted the original data.
    if os.path.isfile(filename):
        shutil.copyfile(filename, filename+"~")
    rnn.save(filename)
    print("Saved trained network to file: ", filename)


def separate(filename, input_files, output_files, extra=None):
    """Use recurrent neural network to separate given input sounds.

    Args:
        filename: File to load neural network from.
        input_files: Input sound files that need to be separated.
        output_files: Files to output separated sounds.
        extra: Extra sounds to plot for comparision.
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
        if len(inputs) == len(rnn.inputs):
            break
        sounds = smanager.read_file(i)
        if type(sounds) == tuple:
            for s in sounds:
                if len(inputs) == len(rnn.inputs):
                    break
                s.sample = s.sample
                inputs.append(list(s.sample))
                rate = s.rate
                inputsounds.append(s)
        else:
            sounds.sample = s.sample
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
    if extra:
        smanager.plot([tuple(inputsounds), tuple(outputs), extra])
    else:
        smanager.plot([tuple(inputsounds), tuple(outputs)])


if __name__ == "__main__":
    sm = sound_manager.SoundManager()

    # filename = "sample_networks/2_speeches.rnn"
    # create_rnn(filename, 2, 20, 2)

    # train_rnn(filename,
    #           input_files=["sound/WAV/X_rss.wav"],
    #           output_files=["sound/WAV/Y1_rss.wav", "sound/WAV/Y2_rss.wav"],
    #           learning_rate=5, iterations=30,
    #           num_samples=10, offset=0, frames=200,
    #           log_file=filename[:-4]+"_log.txt")

    # extra1 = sm.read_file("sound/WAV/Y1_rss.wav")[0]
    # extra2 = sm.read_file("sound/WAV/Y2_rss.wav")[0]
    # separate(filename,
    #          input_files=["sound/WAV/X_rss.wav"],
    #          output_files=["sound/WAV/output1.wav", "sound/WAV/output2.wav"],
    #          extra=(extra1, extra2))

    filename = "sample_networks/first_working.rnn"
    #create_rnn(filename, 2, 40, 1)

    #train_rnn(filename,
    #          input_files=["sound/WAV/X_rsm2.wav"],
    #          output_files=["sound/WAV/Y1_rsm2.wav"],
    #          learning_rate=1, iterations=2000,
    #          num_samples=10, offset=0, frames=200,
    #          log_file=filename[:-4]+"_log.txt")

    extra = sm.read_file("sound/WAV/Y1_rsm2.wav")[0]
    separate(filename,
             input_files=["sound/WAV/X_rsm2.wav"],
             output_files=["sound/WAV/output.wav"],
             extra=extra)
