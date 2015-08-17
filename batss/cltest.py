import pyopencl as cl
import numpy as np
from clw import clwrapper
from clw import kernel


clw = clwrapper.CL()
add = kernel.Kernel(clw)
mult = kernel.Kernel(clw)

add.load('test1.cl')
mult.load('test2.cl')

# h stands for host (CPU)
h_a = np.array(range(10), dtype=np.float32)
h_b = np.array(range(10), dtype=np.float32)

mf = cl.mem_flags

# d stands for device (GPU)
d_a = cl.Buffer(clw.get_context(),
                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(clw.get_context(),
                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(clw.get_context(),
                mf.WRITE_ONLY, h_b.nbytes)
d_d = cl.Buffer(clw.get_context(),
                mf.WRITE_ONLY, h_b.nbytes)

add.execute(h_a.shape, d_a, d_b, d_c)
mult.execute(h_a.shape, d_a, d_b, d_d)

h_c = np.empty_like(h_a)
h_d = np.empty_like(h_a)
cl.enqueue_read_buffer(clw.get_queue(), d_c, h_c).wait()
cl.enqueue_read_buffer(clw.get_queue(), d_d, h_d).wait()
print(h_c)
print(h_d)
