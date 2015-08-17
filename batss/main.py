from rnn import network
from sound import sound, sound_manager
import numpy as np


smanager = sound_manager.SoundManager()
sounds = smanager.read_file("sound/WAV/X_linear.wav")
input1 = list(sounds[0].sample)
input2 = list(sounds[1].sample)
# mix = np.matrix([[0.5, 0.5], [0.5, -0.5]]) * np.matrix([input1, input2])

rnn = network.Network(2, 10, 2)

print(len(input1))

finaloutputs = []
for i in range(int(len(input1)/10)):
    rnn.set_inputs([input1[i*10], input2[i*10]])
    for j in range(10):
        rnn.forward(False)
        rnn.train()
    rnn.forward(True)

out1 = sound.Sound(np.asarray([s[0]*10 for s in rnn.samples]), sounds[0].rate)
out2 = sound.Sound(np.asarray([s[1]*10 for s in rnn.samples]), sounds[0].rate)

smanager.plot(sounds)
smanager.plot((out1, out2))

# Uncomment following to test file loading and saving

# rnn.save("test.rnn")
# neurnn = network.Network.load("test.rnn")
# neurnn.forward()
# print(neurnn.get_outputs())
