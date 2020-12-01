import os
from enum import Enum
import random

import numpy as np
import tensorrt as trt

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
        self.cluster_center = cluster_center

    def load_model(self, target):
        if self.model_engine.get(target, None) is not None:
            return self.model_engine[target]

        model_path = self.models[target]
        engine_file_path = model_path.replace('.onnx', '.trt')
        model_engine = None
        if os.path.exists(engine_file_path):
            print("Reading engine from: ", engine_file_path)
            # deserialize the engine file
            with open(engine_file_path, "rb") as model, trt.Runtime(TRT_LOGGER) as runtime:
                model_engine = runtime.deserialize_cuda_engine(model.read())
        else:
            with trt.Builder(TRT_LOGGER) as builder:
                EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
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
        return self.model_engine[target]

    def load_models_blocking(self, targets):
        engines = {}
        for target in targets:
            engines[target] = self.load_model(target)
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
    def _selected_next(self, cluster_rank):
        for c in cluster_rank:
            cands = self.model_class == c
            for cand in cands:
                name = self.model_arr[cand]
                if self.model_status[name] == ModelStatus.SWAP_OFF:
                    return name
        return None

    def swapoff_and_next(self, removed_one, cluster_rank):
        nxt = self._selected_next(cluster_rank)

        self.model_status[removed_one] = ModelStatus.SWAP_OFF
        del self.model_engine[removed_one]

        return nxt

    def kick_and_next(self, removed_one, cluster_rank):
        nxt = self._selected_next(cluster_rank)

        self.model_status[removed_one] = ModelStatus.KICKED
        del self.model_engine[removed_one]

        return nxt
