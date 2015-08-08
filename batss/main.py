from rnn import network
import sound


rnn = network.Network(2, 20, 2)
rnn.set_inputs([23, 34])
rnn.forward()
print(rnn.get_outputs())

# Uncomment following to test file loading and saving

# rnn.save("test.rnn")
# newrnn = network.Network.load("test.rnn")
# newrnn.forward()
# print(rnn.get_outputs())
