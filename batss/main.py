from rnn import network
import sound
import numpy as np


input1 = [np.sin(i)+np.cos(9*i) for i in range(50)]
input2 = [np.cos(2*i)-5*np.sin(3*i)+10*np.sin(4*i-8) for i in range(50)]
mix = np.matrix([[0.5, 0.5], [0.5, -0.5]]) * np.matrix([input1, input2])

rnn = network.Network(2, 20, 2)
finaloutputs = []
for i in range(50):
    rnn.set_inputs([mix[0,i], mix[1,i]])
    for i in range(50):
        rnn.forward(False)
        rnn.train()
    rnn.forward(True)

print(rnn.samples)

# Uncomment following to test file loading and saving

# rnn.save("test.rnn")
# neurnn = network.Network.load("test.rnn")
# neurnn.forward()
# print(rnn.get_outputs())
