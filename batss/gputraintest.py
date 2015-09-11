import pyopencl as cl
import numpy as np
from clw import clwrapper
from clw import kernel


clw = clwrapper.CL()
forward = kernel.Kernel(clw)
train_pass1 = kernel.Kernel(clw)
train_pass2 = kernel.Kernel(clw)
train_pass3 = kernel.Kernel(clw)

forward.load('forward.cl')
train_pass1.load('train_pass1.cl')
train_pass2.load('train_pass2.cl')
train_pass3.load('train_pass3.cl')


neurons_per_layer = 3
num_layers = 10

io_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)
weight_array = np.array(
    range(neurons_per_layer*neurons_per_layer*num_layers),
    dtype=np.float32)
pot_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)
delta_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)
expected_op_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)

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


for i in range(1, num_layers) :
    forward.execute((neurons_per_layer, ),
                    None,
                    d_io_array,
                    d_weight_array,
                    d_pot_array,
                    np.int32(i*neurons_per_layer),
                    np.int32(neurons_per_layer*neurons_per_layer*i),
                    np.int32(neurons_per_layer),
                    np.int32(neurons_per_layer))

train_pass1.execute((neurons_per_layer, ), 
                    None,
                    )

cl.enqueue_read_buffer(clw.get_queue(), d_io_array, io_array).wait()
print(io_array)
