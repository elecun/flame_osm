import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Engine saved to {engine_path}")
    return True

# Convert
build_engine_from_onnx(
    'yolo11x-pose.onnx',
    'yolo11x-pose.engine'
)