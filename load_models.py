import os
from enum import Enum
import random

import numpy as np
import tensorrt as trt
import time
import onnxruntime as ort

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelStatus(Enum):
    SWAP_OFF = 1
    SWAP_ON = 2
    KICKED = 3


class ModelStore(object):
    # models: {
    #   name1: onnx_path,
    #   name2: onnx_path
    # }
    # models_class: {
    #   name1: 1,
    #   name2: 1,
    #   name3: 2
    #   name4: 2
    # }
    # cluster_center: {
    #   1: name1,
    #   2: name2,
    #   3: name3,
    # }
    def __init__(self, models, models_class, cluster_center):
        self.use_onnx = False
        self.models = models
        self.model_status = {}
        for name in models.keys():
            self.model_status[name] = ModelStatus.SWAP_OFF
        self.model_engine = {}
        # For cluster
        self.model_arr = []
        self.model_class = np.zeros([len(models_class)])
        for i, (name, cluster) in enumerate(models_class.items()):
            self.model_arr.append(name)
            self.model_class[i] = cluster
        self.model_arr = np.array(self.model_arr)
        self.cluster_center = cluster_center

    def load_model(self, target):
        start_t = time.perf_counter()
        if self.model_engine.get(target, None) is not None:
            return self.model_engine[target]

        model_path = self.models[target]
        engine_file_path = model_path.replace('.onnx', '.trt')
        if self.use_onnx:
            model_engine = ort.InferenceSession(model_path)
        else:
            if os.path.exists(engine_file_path):
                print("Reading engine from: ", engine_file_path)
                # deserialize the engine file
                with open(engine_file_path, "rb") as model, trt.Runtime(TRT_LOGGER) as runtime:
                    model_engine = runtime.deserialize_cuda_engine(model.read())
            else:
                with trt.Builder(TRT_LOGGER) as builder:
                    EXPLICIT_BATCH = 1 << (int)(
                        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                    network = builder.create_network(EXPLICIT_BATCH)
                    parser = trt.OnnxParser(network, TRT_LOGGER)
                    builder.max_workspace_size = 1 << 28
                    builder.max_batch_size = 1
                    print(model_path)
                    with open(model_path, 'rb') as onnx_model:
                        parser.parse(onnx_model.read())
                    model_engine = builder.build_cuda_engine(network)
                    with open(engine_file_path, "wb") as f:
                        f.write(model_engine.serialize())

        self.model_engine[target] = model_engine
        self.model_status[target] = ModelStatus.SWAP_ON
        end_t = time.perf_counter()
        return self.model_engine[target], (end_t - start_t)*1000.0

    def load_models_blocking(self, targets):
        engines = {}
        for target in targets:
            engines[target], _ = self.load_model(target)
        return engines

    def select_top_k(self, k):
        out = []
        for _, name in self.cluster_center.items():
            if self.model_status[name] == ModelStatus.SWAP_OFF:
                out.append(name)
            if len(out) == k:
                break
        if len(out) < k:
            all_avail_name = []
            for name, v in self.model_status.items():
                if name not in out and v == ModelStatus.SWAP_OFF:
                    all_avail_name.append(name)
            random.shuffle(all_avail_name)
            out.extend(all_avail_name[:k - len(out)])
        return out

    # Selecting method:
    # Selecting model from the best cluster + probability random select
    def _selected_next(self, cluster_rank, pull_cnt):
        p = random.uniform(0, 1)
        if p < 0.2:
            all_cand = [
                name for name in self.model_arr if self.model_status[name] == ModelStatus.SWAP_OFF]
            if len(all_cand) > 0:
                random.shuffle(all_cand)
                return all_cand[0]
        else:
            for c in cluster_rank:
                cand = []
                for name in self.model_arr[self.model_class == c]:
                    if self.model_status[name] == ModelStatus.SWAP_OFF:
                        cand.append(name)
                if len(cand) > 0:
                    pcnt = np.array([pull_cnt[c] for c in cand])
                    return cand[pcnt.argsort()[0]]
        return None

    def swapoff_and_next(self, removed_one, cluster_rank, pull_cnt):
        nxt = self._selected_next(cluster_rank, pull_cnt)

        self.model_status[removed_one] = ModelStatus.SWAP_OFF
        del self.model_engine[removed_one]
        ms = 0
        if nxt is not None:
            # This should change to be async if use in real world
            _, ms = self.load_model(nxt)
        return nxt, ms

    def kick_and_next(self, removed_one, cluster_rank, pull_cnt):
        nxt = self._selected_next(cluster_rank, pull_cnt)

        self.model_status[removed_one] = ModelStatus.KICKED
        del self.model_engine[removed_one]
        ms = 0
        if nxt is not None:
            # This should change to be async if use in real world
            _, ms = self.load_model(nxt)
        return nxt, ms
