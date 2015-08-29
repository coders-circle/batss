import pyopencl as cl
import numpy as np
from clw import clwrapper
from clw import kernel


clw = clwrapper.CL()
#add = kernel.Kernel(clw)
#mult = kernel.Kernel(clw)
matmult = kernel.Kernel(clw)

#add.load('test1.cl')
#mult.load('test2.cl')
matmult.load('matmult.cl')


rowsA = 1000
colsA = 1000
rowsB = 1000
colsB = 1000

# h stands for host (CPU)
#h_a = np.array(range(10), dtype=np.float32)
#h_b = np.array(range(10), dtype=np.float32)
h_matA = np.array(range(rowsA*colsA), dtype=np.float32)
h_matB = np.array(range(rowsB*colsB), dtype=np.float32)
h_matC = np.empty_like(h_matA)

mf = cl.mem_flags

# d stands for device (GPU)
#d_a = cl.Buffer(clw.get_context(),
#                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
#d_b = cl.Buffer(clw.get_context(),
#                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
#d_c = cl.Buffer(clw.get_context(),
#                mf.WRITE_ONLY, h_b.nbytes)
#d_d = cl.Buffer(clw.get_context(),
#                mf.WRITE_ONLY, h_b.nbytes)

d_matA = cl.Buffer(clw.get_context(),
                   mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_matA)
d_matB = cl.Buffer(clw.get_context(),
                   mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_matB)
d_matC = cl.Buffer(clw.get_context(), 
                   mf.WRITE_ONLY, h_matC.nbytes)


#add.execute(h_a.shape, d_a, d_b, d_c)
#mult.execute(h_a.shape, d_a, d_b, d_d)
matmult.execute((rowsA, colsB), None, d_matC, d_matA, d_matB, np.int32(colsA), np.int32(colsB))

#h_c = np.empty_like(h_a)
#h_d = np.empty_like(h_a)
#cl.enqueue_read_buffer(clw.get_queue(), d_c, h_c).wait()
#cl.enqueue_read_buffer(clw.get_queue(), d_d, h_d).wait()
#print(h_c)
#print(h_d)

cl.enqueue_read_buffer(clw.get_queue(), d_matC, h_matC).wait()
print(h_matC)

h_matD = np.dot(np.resize(h_matA, (rowsA, colsA)), np.resize(h_matB, (rowsB, colsB)))
print(np.resize(h_matD, rowsA*colsA))