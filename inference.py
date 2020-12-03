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

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# class_idx = json.load(open("labels.json"))
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def allocate_buffers(model_engine):
    bindings = []
    inputs = []
    outputs = []
    # binding: describe the input and output ports of the engine
    for binding in model_engine:
        data_size = trt.volume(model_engine.get_binding_shape(
            binding)) * model_engine.max_batch_size
        data_type = trt.nptype(model_engine.get_binding_dtype(binding))
        host_memory = pycuda.driver.pagelocked_empty(data_size, data_type)
        device_memory = pycuda.driver.mem_alloc(host_memory.nbytes)
        # stored the memory index in CUDA context
        bindings.append(int(device_memory))
        if model_engine.binding_is_input(binding):
            inputs.append({"host": host_memory, "device": device_memory})
        else:
            outputs.append({"host": host_memory, "device": device_memory})
    return inputs, outputs, bindings


def do_inference(context, bindings, inputs, outputs, stream):
    start = time.perf_counter()
    # send inputs to device (GPU)
    for input in inputs:
        pycuda.driver.memcpy_htod_async(input["device"], input["host"], stream)
    # do inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # send outputs to host (CPU)
    for output in outputs:
        pycuda.driver.memcpy_dtoh_async(
            output["host"], output["device"], stream)
    # waot for all activity on this stream to cease, then return.
    stream.synchronize()
    end = time.perf_counter()
    return [output["host"] for output in outputs], (end - start) * 1000.0


"""
Using the groundtruth as the user/system feedback
"""


def reward_fn_feedback(predict, gt):
    if predict == gt:
        return 1.0
    else:
        return 0


"""
Using the weighted accuracy of the expected accuracy as reward
"""


def reward_fn_distri(predict, expected_accuracy):
    n_label = len(expected_accuracy)
    if predict >= n_label:
        return None
    return expected_accuracy[predict]


def main():
    K = 2
    WINDOW_SIZE = 1000
    EXPLORE_THRESHOLD = 1000
    FPS = 30.0
    OMSI_CONF = './omsi_conf.yaml'
    DATASET_DIR = './tiny-imagenet-200/train'
    PROBABILITY_SEQ = 'PSeq1'

    # Step 1: Load the config: List of models, expected accuracy on each class, model distance
    #     Step 1.1: Init to Select K first cand models
    omsi = omsi_loader.OMSILoader()
    omsi.load(OMSI_CONF)
    store = omsi.model_store()
    k_models = store.select_top_k(K)
    agent = NonStationaryBanditAgent(Policy(), K, k_models)

    # Step 2: Create context for data pipeline
    data_itr = dataloader.load_with_probability_seq(
        PROBABILITY_SEQ, DATASET_DIR)

    # Step 3: Create engine and context for the K cand models
    engines_map = store.load_models_blocking(k_models)
    engines = [engines_map[m] for m in k_models]

    inputs, outputs, bindings = allocate_buffers(engines[0])
    contexts = [engine.create_execution_context() for engine in engines]

    # Step 4: Start N round, inference and profile
    #     Step 4.1: Inference
    #     Step 4.2: Time as a hard constraint, replace the failed models (failed fps with n out of m time in window)
    #               (First find best predict acc in group, Otherwise, use best predict acc in other group)
    #     Step 4.3: Sliding window for class label
    #     Step 4.4: Update predict acc
    profiler = profiling.Profiler(WINDOW_SIZE, FPS, omsi.model_cluster())
    stream = pycuda.driver.Stream()
    for img, gt in data_itr:
        target_idx = agent.choose()
        target_model = k_models[target_idx]
        target_context = contexts[target_idx]
        inputs[0]["host"] = np.array(img, dtype=np.float32, order='C')
        out, ms = do_inference(target_context, bindings,
                               inputs, outputs, stream)

        # Convert the 1000 dimension output to label
        trt_output = torch.nn.functional.softmax(torch.Tensor(out[0]), dim=0)
        label = trt_output.argmax(dim=0).numpy()

        reward = reward_fn_feedback(int(label), int(gt))
        # reward = reward_fn_distri(label, omsi.expected_accuracy(target_model))

        # Unavailable to evaluate
        if reward is None:
            continue

        agent.observe(target_model, reward)
        agent.set_explore_threshold(EXPLORE_THRESHOLD)
        profiler.profile_once(target_model, reward, ms)

        # Replace model in K cand set
        if agent.initialized():
            if not profiler.model_acceptable(target_model):
                print("Unacceptable %s" % target_model)
                # print(profiler.selected_record)
                profiler.invalid_model(target_model)
                agent.kick_option(target_model)
                new_target = store.kick_and_next(
                    target_model, profiler.cluster_rank(),
                    profiler.model_pull_cnt)
                if new_target is not None:
                    agent.add_option(new_target)
                    k_models[target_idx] = new_target

        worst_cand = agent.vote_the_worst()
        if worst_cand is not None:
            print("Swap off the not potential")
            # profiler.show_mean_score()
            w_model = k_models[worst_cand]
            # print(profiler.selected_record)
            agent.kick_option(w_model)
            new_target = store.swapoff_and_next(
                w_model, profiler.cluster_rank(),
                profiler.model_pull_cnt)
            if new_target is not None:
                agent.add_option(new_target)
                k_models[worst_cand] = new_target

    torch.cuda.empty_cache()
    # Step N: Plot the model selection flow, accuracy flow, fps flow
    profiler.draw_report()


if __name__ == '__main__':
    main()
