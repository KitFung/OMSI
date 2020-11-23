import time
import os

import torchvision
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda
import pycuda.autoinit
import torch
import numpy as np
from PIL import Image
import json

from bandit import NonStationaryBanditAgent, Policy
import omsi_loader
import load_models
import dataloader
import profiling

class_idx = json.load(open("labels.json"))
onnx_file_path = 'vgg.onnx'
engine_file_path = 'vgg.trt'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# def get_image(input_image_path):
# 	print("Get image: ", input_image_path)
# 	image = Image.open(input_image_path)
# 	print("Input image format {}, size {}, mode {}.".format(image.format, image.size, image.mode))
# 	preprocess = transforms.Compose([
# 		transforms.Resize(256),
# 		transforms.CenterCrop(224),
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# 	])
# 	image = preprocess(image)
# 	print("Image size after preprocessing: ", image.shape)
# 	image_binary = np.array(image, dtype=np.float32, order='C')
# 	return image_binary

# def load_model():
#     if not os.path.exists(onnx_file_path):
#         dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
#         model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13', pretrained=True).cuda()
#         # names is just for readaility, not name the weight manually
#         input_names = ["actual_input_1"]
#         output_names = ["output1"]
#         torch.onnx.export(model, dummy_input, onnx_file_path, verbose=True, input_names=input_names, output_names=output_names)
#     onnx_model = open(onnx_file_path, 'rb')
#     return onnx_model


def allocate_buffers(model_engine):
	bindings 	= []
	inputs 		= []
	outputs 	= []
	# binding: describe the input and output ports of the engine
	for binding in model_engine:
		data_size 		= trt.volume(model_engine.get_binding_shape(binding)) * model_engine.max_batch_size
		data_type 		= trt.nptype(model_engine.get_binding_dtype(binding))
		host_memory 	= pycuda.driver.pagelocked_empty(data_size, data_type)
		device_memory 	= pycuda.driver.mem_alloc(host_memory.nbytes)
		# stored the memory index in CUDA context
		bindings.append(int(device_memory))
		if model_engine.binding_is_input(binding):
			inputs.append({"host": host_memory, "device": device_memory})
		else:
			outputs.append({"host": host_memory, "device": device_memory})
	return inputs, outputs, bindings

# def create_model_engine(engine_file_path, onnx_model):
#     model_engine = None
#     if os.path.exists(engine_file_path):
#         print("Reading engine from: ", engine_file_path)
#         # deserialize the engine file
#         with open(engine_file_path, "rb") as model, trt.Runtime(TRT_LOGGER) as runtime:
#             model_engine = runtime.deserialize_cuda_engine(model.read())
#     else:
#         with trt.Builder(TRT_LOGGER) as builder:
#             # Specify that the network should be created with an explicit batch dimension
#             EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#             network = builder.create_network(EXPLICIT_BATCH)
#             parser = trt.OnnxParser(network, TRT_LOGGER)
#             builder.max_workspace_size = 1 << 28
#             builder.max_batch_size = 1
#             parser.parse(onnx_model.read())
#             model_engine = builder.build_cuda_engine(network)
#             with open(engine_file_path, "wb") as f:
#                 f.write(model_engine.serialize())
#     return model_engine

def do_inference(context, bindings, inputs, outputs, stream):
	start = time.perf_counter_ns()
	# send inputs to device (GPU)
	for input in inputs:
		pycuda.driver.memcpy_htod_async(input["device"], input["host"], stream)
	# do inference
	context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
	# send outputs to host (CPU)
	for output in outputs:
		pycuda.driver.memcpy_dtoh_async(output["host"], output["device"], stream)
	# waot for all activity on this stream to cease, then return.
	stream.synchronize()
	end = time.perf_counter_ns()
	return [output["host"] for output in outputs], (end - start) * 1e-6

 

def main():
    K = 3
    WINDOW_SIZE = 1000
    OMSI_CONF = './omsi_conf.yaml'

    # Step 1: Load the config: List of models, expected accuracy on each class, model distance
    #     Step 1.1: Init to Select K first cand models
    osmi = omsi_loader.OMSILoader()
    osmi.load(OMSI_CONF)
    store = osmi.model_store()
    k_models = store.select_top_k(K)

    agent = NonStationaryBanditAgent(Policy(), K, k_names)

    # Step 2: Create context for data pipeline
    data_itr = dataloader.load_with_probability_seq("PSeq1")
    # Step 3: Create engine and context for the K cand models
    models = store.load_models_blocking(k_models)

    # Step 4: Start N round, inference and profile
    #     Step 4.1: Inference
    #     Step 4.2: Time as a hard constraint, replace the failed models (failed fps with n out of m time in window)
    #               (First find best predict acc in group, Otherwise, use best predict acc in other group)
    #     Step 4.3: Sliding window for class label 
    #     Step 4.4: Update predict acc
    profiler = profiling.Profiler(WINDOW_SIZE)
    for data in data_itr:
        target_model = agent.choose()

        out, profilems = do_inference()

        agent.observe()

        profiler.profile_time(profilems)
        profiler.profile_label(out)

        # update k cand models list:
        if not profiler.model_acceptable(target_model):
            agent.kick_option()
            new_target = selector.selected_next(self, idx)
            agent.add_option(new_target)

    # Step N: Plot the model selection flow, accuracy flow, fps flow
    profiler.draw_report()

if __name__ == '__main__':
    main()
    # model = load_model()
    # model_engine = create_model_engine(engine_file_path, model)

    # with model_engine.create_execution_context() as context:
    #     inputs, outputs, bindings = allocate_buffers(model_engine)
    #     stream = pycuda.driver.Stream()
    #     inputs[0]["host"] = image
    #     outputs, trt_time = do_inference(context, bindings, inputs, outputs, stream)

    #     trt_output = torch.nn.functional.softmax(torch.Tensor(outputs[0]), dim=0)
    #     label = trt_output.argmax(dim=0).numpy()
    #     print("trt_time:     %.6f seconds." % trt_time)
    #     print("trt_label:   ", label)
    #     print("Class: %s" % class_idx[str(label)][1])
    # torch.cuda.empty_cache()
