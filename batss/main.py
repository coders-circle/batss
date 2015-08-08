from rnn import network
import sound


rnn = network.Network(2, 20, 2)
rnn.set_inputs([23, 34])
rnn.forward()
print(rnn.get_outputs())
