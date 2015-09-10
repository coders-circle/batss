from sound import sound, sound_manager
import numpy as np
from backprop import network


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


sm = sound_manager.SoundManager()
# input0 = sm.read_file("sound/WAV/X_rss.wav")[0]
input0 = sm.read_file("sound/WAV/X_rsm2.wav")[0]

# target0 = sm.read_file("sound/WAV/Y1_rss.wav")[0]
target0 = sm.read_file("sound/WAV/Y1_rsm2.wav")[0]

samples = 1000
rnn = network.Network(samples, [samples], samples)
rate = 0.03

for k in range(20):
    print("Iteration #", k, "Learning Rate:", rate)
    errs = []
    for j in range(int(len(target0.sample)/samples/2)):
        off = j * int(samples/2)
        ins = input0.sample[off:off+int(samples/2)]
        outs = target0.sample[off:off+int(samples/2)]
        iamp, iarg = sm.fft(ins)
        iamp /= 1000
        iarg /= 1000
        oamp, oarg = sm.fft(outs)
        oamp /= 1000
        oarg /= 1000
        ins = list(iamp + iarg)
        outs = list(oamp + oarg)
        e = rnn.train(ins, outs, rate)
        errs.append(e)
    
outputs = []
for i in range(int(len(input0.sample)/samples)):
    off = i * samples
    ins = input0.sample[off:off+samples]
    iamp, iarg = sm.fft(ins)
    iamp /= 1000
    iarg /= 1000
    ins = list(iamp[:samples] + iarg[:samples])
    rnn.set_inputs(ins)
    rnn.forward()
    outs = chunkify(list(np.array(rnn.get_outputs()) * 1000), 2)
    outs = sm.ifft(np.array(outs[0]), np.array(outs[1]))
    outputs.append(list(outs))

osounds = sum(outputs, [])
osounds = sound.Sound(np.array(osounds), input0.rate)
sm.plot_fft([input0, osounds, target0])

sm.save_file(osounds, "sound/WAV/output.wav")
