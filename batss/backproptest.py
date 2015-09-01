from sound import sound, sound_manager
import numpy as np
from backprop import network


sm = sound_manager.SoundManager()
# input0 = sm.read_file("sound/WAV/X_rss.wav")[0]
input0 = sm.read_file("sound/WAV/X_rsm2.wav")[0]

# target0 = sm.read_file("sound/WAV/Y1_rss.wav")[0]
target0 = sm.read_file("sound/WAV/Y1_rsm2.wav")[0]

samples = 1000
rnn = network.Network(samples, samples, samples)
rate = 0.03

for k in range(30):
    print("Iteration #", k, "Learning Rate:", rate)
    for j in range(int(len(target0.sample)/samples)):
        off = j * samples
        e = rnn.train(input0.sample[off:off+samples], target0.sample[off:off+samples], rate)
        errs.append(e)
    
outputs = []
for i in range(int(len(input0.sample)/samples)):
    off = i * samples
    rnn.set_inputs(input0.sample[off:off+samples])
    rnn.forward()
    outputs.append(rnn.get_outputs())

osounds = sum(outputs, [])
osounds = sound.Sound(np.array(osounds), input0.rate)
sm.plot([input0, osounds, target0])

sm.save_file(osounds, "sound/WAV/output.wav")
