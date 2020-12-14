import time
import collections
import os
import sys
import warnings
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
import model_store
import dataloader
import profiling
import metric


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


def pre_measure_inference_time(img, store, models):
    times = {}
    stream = pycuda.driver.Stream()
    for model in models:
        engine, _ = store.load_model(model)
        inputs, outputs, bindings = allocate_buffers(engine)
        inputs[0]["host"] = np.array(img, dtype=np.float32, order='C')
        context = engine.create_execution_context()
        for _ in range(5):
            _, ms = do_inference(context, bindings, inputs, outputs, stream)
            times[model] = ms
        store.removed_model(model)
    return times


def main():
    K = 5
    WINDOW_SIZE = 1000
    EXPLORE_THRESHOLD = 1000
    FPS = 30.0
    WEIGHT_UPDATE_WINDOW = 256
    OMSI_CONF = './omsi_conf.yaml'
    DATASET_DIR = './tiny-imagenet-200/train'

    if len(sys.argv) > 1:
        OMSI_CONF = sys.argv[1]
    if len(sys.argv) > 2:
        K = int(sys.argv[2])
    if len(sys.argv) > 3:
        DATASET_DIR = sys.argv[3]

    PROBABILITY_SEQ = 'PSeq1'

    # Step 1: Create context for data pipeline
    data_itr = dataloader.load_with_probability_seq(
        PROBABILITY_SEQ, DATASET_DIR)

    # Step 2: Load the config: List of models, expected accuracy on each class, model distance
    #     Step 1.1: Init to Select K first cand models
    omsi = omsi_loader.OMSILoader()
    omsi.load(OMSI_CONF)
    store = omsi.model_store()

    expected_inference_time = pre_measure_inference_time(
        next(data_itr)[0], store, store.model_arr)
    ave_inference_time = np.mean(list(expected_inference_time.values()))

    K = min(len(store.model_arr), int((1000.0 / FPS) / ave_inference_time))
    print(">> K: %d" % K)
    k_models = store.select_top_k(K)
    agent = NonStationaryBanditAgent(
        Policy(), len(store.model_arr), store.model_arr)
    agent.set_explore_threshold(EXPLORE_THRESHOLD)

    # Step 3: Create engine and context for the K cand models
    engines_map = store.load_models_blocking(k_models)
    engines = [engines_map[m] for m in k_models]
    inputs, outputs, bindings = allocate_buffers(engines[0])
    contexts = [engine.create_execution_context() for engine in engines]

    profiler = profiling.Profiler(WINDOW_SIZE, FPS, omsi.model_cluster())
    stream = pycuda.driver.Stream()
    count_itr = 0
    start_t = time.perf_counter()

    softmax_chunk = collections.defaultdict(list)
    gt_chunk = []
    weights = np.zeros(K)
    weights[:] = 1.0 / K
    total_acc = 0

    for img, gt in data_itr:
        start_iter_t = time.perf_counter()
        count_itr += 1
        load_model_ms = 0.0  # for time load_model

        # Step 4: Update the weight
        if count_itr % WEIGHT_UPDATE_WINDOW == 0:
            mse_arr = np.array(
                [metric.MSE(softmax_chunk[m], gt_chunk) for m in k_models])
            new_weights = np.array([metric.mse2weight(mse) for mse in mse_arr])

            softmax_chunk.clear()
            gt_chunk.clear()
            # print(k_models)
            # print(new_weights)
            # print('-----------')
            # Check whether need update model set
            if agent.explore_enough(k_models):
                nxt_model = agent.choose()
                if nxt_model not in k_models:
                    print("Swap off the not potential at round %d" % count_itr)
                    worst_ind = new_weights.argmin()
                    w_model = k_models[worst_ind]
                    _, load_model_ms = store.swapoff_and_custom_next(
                        w_model, nxt_model)
                    k_models[worst_ind] = nxt_model
            weights = new_weights

        # Step 5: Inference with ensemble
        label_comb = None
        inputs[0]["host"] = np.array(img, dtype=np.float32, order='C')
        inference_ms = 0
        for idx in range(K):
            target_model = k_models[idx]
            out, ms = do_inference(contexts[idx], bindings,
                                   inputs, outputs, stream)
            inference_ms += ms
            # Convert the 1000 dimension output to label
            trt_output = torch.nn.functional.softmax(
                torch.Tensor(out[0]), dim=0)
            output_np = trt_output.numpy()
            if label_comb is None:
                label_comb = weights[idx] * output_np
            else:
                label_comb += weights[idx] * output_np

            # Update single model stat
            label = trt_output.argmax(dim=0).numpy()
            model_reward = metric.reward_fn_feedback(int(label), int(gt))
            agent.observe(target_model, model_reward)
            # Update chunk observe
            softmax_chunk[target_model].append(output_np)
        gt_chunk.append(int(gt))

        # Step 6: Update and record from the result

        # Update the chunk
        ensemble_label = label_comb.argmax()
        ensemble_reward = metric.reward_fn_feedback(
            int(ensemble_label), int(gt))

        # Update profiler
        profiler.log_load_model(load_model_ms)
        profiler.profile_once(k_models, weights, ensemble_reward, inference_ms)

        end_iter_t = time.perf_counter()
        profiler.log_single_iter((end_iter_t - start_iter_t)*1000.0)
    end_t = time.perf_counter()

    print('-------')
    print('inference image count', count_itr)
    print('[inference_all], elapsed time (s): ', round(end_t - start_t, 4))

    torch.cuda.empty_cache()
    # Step N: Plot result
    profiler.export_json()
    # print_summary
    print('-------')
    print('K:', K)
    print('WINDOW_SIZE:', WINDOW_SIZE)
    print('EXPLORE_THRESHOLD', EXPLORE_THRESHOLD)
    print('FPS', FPS)
    print("Overall Score: %f " % (profiler.overall_score / float(count_itr)))


if __name__ == '__main__':
    main()
