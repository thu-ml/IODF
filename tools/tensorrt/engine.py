import tensorrt as trt
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def GiB(val):
    return val * 1<<30

def build_engine(max_batch_size, fp16_mode, int8_mode, onnx_file_path, calib=None):
    # network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) | \
                # 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # builder.create_network(network_creation_flag) as network,
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(network_creation_flag) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            builder.create_builder_config() as config:

        config.max_workspace_size = GiB(15)
        builder.max_batch_size = max_batch_size
        if int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)
            if calib is not None:
                config.int8_calibrator = calib
        elif fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            pass
        print('Loading ONNX model from {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
            assert network.num_layers > 0, 'Failed to parse ONNX model. Please check if the ONNX model is compatible '
        # last_layer = network.get_layer(network.num_layers - 1)
        # network.mark_output(last_layer.get_output(0))
        print('Completed parsing of ONNX file, network has {} layers.'.format(network.num_layers))
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))    
        engine = builder.build_engine(network, config)
    assert engine is not None, 'Failed to create the engine.'
    return engine

def save_engine(engine, fp):
    with open(fp, 'wb') as f:
        f.write(engine.serialize())
        
def load_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

