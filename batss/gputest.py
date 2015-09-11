import pyopencl as cl
import numpy as np
from clw import clwrapper
from clw import kernel


clw = clwrapper.CL()
forward = kernel.Kernel(clw)

forward.load('forward.cl')

neurons_per_layer = 3
num_layers = 10

io_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)
weight_array = np.array(
    range(neurons_per_layer*neurons_per_layer*num_layers),
    dtype=np.float32)
pot_array = np.array(range(neurons_per_layer*num_layers), dtype=np.float32)

mf = cl.mem_flags

d_io_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=io_array)
d_weight_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=weight_array)
d_pot_array = cl.Buffer(clw.get_context(),
                   mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pot_array)


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

cl.enqueue_read_buffer(clw.get_queue(), d_io_array, io_array).wait()
print(io_array)