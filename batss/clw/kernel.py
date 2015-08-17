import pyopencl
from clw import clwrapper


class Kernel:
    def __init__(self, _CL):
        self.CL = _CL

    def load(self, filename):
        file = open(filename, 'r')
        fstr = "".join(file.readlines())
        self.program = pyopencl.Program(self.CL.get_context(), fstr).build()

    def execute(self, numthreads, *vars):
        self.program.main(self.CL.get_queue(), numthreads, None, *vars)
