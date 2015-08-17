import pyopencl

# OpenCL Wrapper class
class CL:
    # initialize context and command queue
    def __init__(self):
        self.context = pyopencl.create_some_context()
        self.queue = pyopencl.CommandQueue(self.context)    

    def get_context(self):
        return self.context

    def get_queue(self):
        return self.queue
