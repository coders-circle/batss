import random
from timeit import default_timer as timer
import pyopencl as cl
import numpy as np
from clw import clwrapper
from clw import kernel

from sound import sound, sound_manager
import numpy as np
from backprop import network


def random_weight(div_factor=1):
    #return (random.random() - 0.5) / (div_factor)
    return 1


clw = clwrapper.CL()
forward = kernel.Kernel(clw)
train_pass1 = kernel.Kernel(clw)
train_pass2 = kernel.Kernel(clw)
train_pass3 = kernel.Kernel(clw)

forward.load('forward.cl')
train_pass1.load('train_pass1.cl')
train_pass2.load('train_pass2.cl')
train_pass3.load('train_pass3.cl')


neurons_per_layer = 1000;
num_layers = 3;

io_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)
for i in range(len(io_array)):
    io_array[i] = (io_array[i]%1000)/1000;

weight_array = np.array(
    range(neurons_per_layer*neurons_per_layer*num_layers),
    dtype=np.float32)
for i in range(len(weight_array)):
    weight_array[i] = np.float32(random_weight(len(weight_array)))
#print(weight_array)
pot_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)
delta_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)
expected_op_array = np.array(range(neurons_per_layer-1, -1, -1), dtype=np.float32)
for i in range(len(expected_op_array)):
    expected_op_array[i] /= 10000;

#print(expected_op_array)


mf = cl.mem_flags

d_io_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=io_array)
d_weight_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=weight_array)
d_pot_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pot_array)
d_delta_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=delta_array)
d_expected_op_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=expected_op_array)



sm = sound_manager.SoundManager()
# input0 = sm.read_file("sound/WAV/X_rss.wav")[0]
input0 = sm.read_file("sound/WAV/X_rsm2.wav")[0]

# target0 = sm.read_file("sound/WAV/Y1_rss.wav")[0]
target0 = sm.read_file("sound/WAV/Y1_rsm2.wav")[0]

samples = 1000
num_frame = int(len(target0.sample)/samples)

for k in range(30):
    for j in range(int(len(target0.sample)/samples)):
        off = j * samples
        ins = input0.sample[off:off+samples]
        ins = [np.float32(x) for x in ins]
        outs = target0.sample[off:off+samples]
        outs = [np.float32(x) for x in outs]
        cl.enqueue_write_buffer(clw.get_queue(), mem=d_io_array, hostbuf=np.asarray(ins)).wait()
        cl.enqueue_write_buffer(clw.get_queue(), mem=d_expected_op_array, hostbuf=np.asarray(outs)).wait()
        # forward pass
        for i in range(1, num_layers) :
            forward.execute((neurons_per_layer, ),
                            None,
                            d_io_array,
                            d_weight_array,
                            d_pot_array,
                            np.int32(i*neurons_per_layer),
                            np.int32(neurons_per_layer*neurons_per_layer*(i-1)),
                            np.int32(neurons_per_layer),
                            np.int32(neurons_per_layer))
        # calculate delta of output layer
        train_pass1.execute((neurons_per_layer, ),
                            None,
                            d_expected_op_array,
                            d_io_array,
                            d_delta_array,
                            np.int32(neurons_per_layer*(num_layers-1)))
        # calculate delta of hidden layers
        for i in range(num_layers-2, 0, -1):
            train_pass2.execute((neurons_per_layer, ),
                                None,
                                d_weight_array,
                                d_pot_array,
                                d_delta_array,
                                np.int32((i-1)*neurons_per_layer),
                                np.int32(i*neurons_per_layer),
                                np.int32(neurons_per_layer*neurons_per_layer*(i-1)),
                                np.int32(neurons_per_layer),
                                np.int32(neurons_per_layer))
        # adjust weights according to calculated error
        for i in range(0, num_layers):
            train_pass3.execute((neurons_per_layer,),
                                None,
                                d_io_array,
                                d_weight_array,
                                d_delta_array,
                                np.int32(i*neurons_per_layer),
                                np.int32(neurons_per_layer),
                                np.int32((i+1)*neurons_per_layer),
                                np.int32(neurons_per_layer*neurons_per_layer*i),
                                np.float32(0.01))

outputs = []
for i in range(int(len(input0.sample)/samples)):
    off = i * samples
    ins = input0.sample[off:off+samples]
    ins = [np.float32(x) for x in ins]
    cl.enqueue_write_buffer(clw.get_queue(), mem=d_io_array, hostbuf=np.asarray(ins)).wait()
    for i in range(1, num_layers) :
        forward.execute((neurons_per_layer, ),
                        None,
                        d_io_array,
                        d_weight_array,
                        d_pot_array,
                        np.int32(i*neurons_per_layer),
                        np.int32(neurons_per_layer*neurons_per_layer*(i-1)),
                        np.int32(neurons_per_layer),
                        np.int32(neurons_per_layer))
    cl.enqueue_read_buffer(clw.get_queue(), d_io_array, io_array).wait()
    outputs.append(io_array[neurons_per_layer*(num_layers-1)])


    
osounds = sum(outputs, [])
osounds = sound.Sound(np.array(osounds), input0.rate)
sm.plot([input0, osounds, target0])

sm.save_file(osounds, "sound/WAV/output.wav")




#num_iterations = 100
#start = timer()
#for iteration in range(num_iterations):
#    # forward pass
#    for i in range(1, num_layers) :
#        forward.execute((neurons_per_layer, ),
#                        None,
#                        d_io_array,
#                        d_weight_array,
#                        d_pot_array,
#                        np.int32(i*neurons_per_layer),
#                        np.int32(neurons_per_layer*neurons_per_layer*(i-1)),
#                        np.int32(neurons_per_layer),
#                        np.int32(neurons_per_layer))
#    # calculate delta of output layer
#    train_pass1.execute((neurons_per_layer, ),
#                        None,
#                        d_expected_op_array,
#                        d_io_array,
#                        d_delta_array,
#                        np.int32(neurons_per_layer*(num_layers-1)))
#    # calculate delta of hidden layers
#    for i in range(num_layers-2, 0, -1):
#        train_pass2.execute((neurons_per_layer, ),
#                            None,
#                            d_weight_array,
#                            d_pot_array,
#                            d_delta_array,
#                            np.int32((i-1)*neurons_per_layer),
#                            np.int32(i*neurons_per_layer),
#                            np.int32(neurons_per_layer*neurons_per_layer*(i-1)),
#                            np.int32(neurons_per_layer),
#                            np.int32(neurons_per_layer))
#    # adjust weights according to calculated error
#    for i in range(0, num_layers):
#        train_pass3.execute((neurons_per_layer,),
#                            None,
#                            d_io_array,
#                            d_weight_array,
#                            d_delta_array,
#                            np.int32(i*neurons_per_layer),
#                            np.int32(neurons_per_layer),
#                            np.int32((i+1)*neurons_per_layer),
#                            np.int32(neurons_per_layer*neurons_per_layer*i),
#                            np.float32(0.01))
#    #cl.enqueue_read_buffer(clw.get_queue(), d_io_array, io_array).wait()
#    #cl.enqueue_read_buffer(clw.get_queue(), d_delta_array, delta_array).wait()
#    #print(delta_array)
#    #print(io_array)
#    #print("")


#for i in range(1, num_layers) :
#        forward.execute((neurons_per_layer, ),
#                        None,
#                        d_io_array,
#                        d_weight_array,
#                        d_pot_array,
#                        np.int32(i*neurons_per_layer),
#                        np.int32(neurons_per_layer*neurons_per_layer*(i-1)),
#                        np.int32(neurons_per_layer),
#                        np.int32(neurons_per_layer))

#cl.enqueue_read_buffer(clw.get_queue(), d_io_array, io_array).wait()
#cl.enqueue_read_buffer(clw.get_queue(), d_weight_array, weight_array).wait()

#elapsed_time = timer() - start
#print(elapsed_time)

#print(io_array)
