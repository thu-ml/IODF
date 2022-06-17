import time

import pycuda.autoinit
import pycuda.driver as cuda

def do_inference(context, bindings, inputs, outputs, batch_size=1, timing=False):
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
    t = None
    # start = cuda.Event()
    # end = cuda.Event()
    # tic = time.time()
    # start.record()
    context.execute(batch_size=batch_size, bindings=bindings)
    # end.record()
    # end.synchronize()
    # t = time.time() - tic
    # t = start.time_till(end)*1e-3
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
    if timing:
        return [out.host for out in outputs], t
    else:
        return  [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_async(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]