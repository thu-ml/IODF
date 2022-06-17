import tensorrt as trt
import sys
sys.path.append('..')
import logging

class MyProfiler(trt.IProfiler):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.time = 0.
        # ops = ['quant', 'dequant', 'conv']
        # self.timer = Timer(ops)

    def report_layer_time(self, layer_name, ms):
        # print(f"{layer_name}, {ms}ms")
        self.time += ms/1e3