from rnn import network
from sound import sound, sound_manager
import numpy as np


smanager = sound_manager.SoundManager()
sounds = smanager.read_file("sound/WAV/X_rss.wav")
input1 = list(sounds[0].sample)
input2 = list(sounds[1].sample)

target1 = list(smanager.read_file("sound/WAV/Y1_rss.wav")[0].sample)
target2 = list(smanager.read_file("sound/WAV/Y2_rss.wav")[0].sample)
# mix = np.matrix([[0.5, 0.5], [0.5, -0.5]]) * np.matrix([input1, input2])

rnn = network.Network(2, 10, 2)

ts = 2000  # number of training samples
in1 = input1[0:ts]
in2 = input2[0:ts]
t1 = target1[0:ts]
t2 = target2[0:ts]

print("Number of samples: ", len(input1))
print("Number of traning samples: ", len(in1))
print("Training...")

input_series = [[in1[i], in2[i]] for i in range(len(in1))]
output_series = [[t1[i], t2[i]] for i in range(len(t1))]
for i in range(1):
    rnn.train(input_series, 0.8, output_series)

print("Trained")
print("Separating...")

rnn.samples = []
for i in range(len(input1)):
    rnn.set_inputs([input1[i], input2[i]])
    rnn.forward()

print("Separated")

out1 = sound.Sound(np.asarray([s[0] for s in rnn.samples]), sounds[0].rate)
out2 = sound.Sound(np.asarray([s[1] for s in rnn.samples]), sounds[0].rate)

out1.reshape()
out2.reshape()

smanager.plot([sounds, (out1, out2)])

smanager.save_file(out1, "sound/WAV/output1.wav")
smanager.save_file(out2, "sound/WAV/output2.wav")

# Uncomment following to test file loading and saving

# rnn.save("test.rnn")
# neurnn = network.Network.load("test.rnn")
# neurnn.forward()
# print(neurnn.get_outputs())
