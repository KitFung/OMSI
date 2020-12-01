import os
import sys
import yaml

import numpy as np
from sklearn.cluster import KMeans
import torch
from ptflops import get_model_complexity_info
import pretrainedmodels

input_shape = (3, 416, 416)

# Do the clustering on torch model
# # man craft feature: [sparity, number of fullcon, number of dropout, number of conv, number of activate, number of normalization, depth]
# man craft feature: [macs, params, sparity]
def gen_feature(model, sparisity):
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    macs = float(macs.split()[0])
    params = float(params.split()[0])
    return (macs, params, sparisity)

# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/Swall0w/torchstat
# https://github.com/sovrasov/flops-counter.pytorch

def gen_features(models):
    return {name: gen_feature(model, sparisity) for name, (model, sparisity) in models.items()}

# models: {
#   model_a: feature vector,
#   model_b: feature vector,
#   model_c: feature vector,
#   model_d: feature vector,
# }
def clustering(features):
    names = list(features.keys())
    X = np.array([list(features[k]) for k in names])
    X_normed = X / X.max(axis=0)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_normed)
    out = {names[i]: int(kmeans.labels_[i]) for i in range(len(names))}
    return out

# ## INPUT YAML
def load_models(conf_file):
    out = {}
    with open(conf_file) as f:
        doc = yaml.full_load(f)
    models = doc['config']
    for k, stat in models.items():
        if stat['net'] == 'inception_v4':
            out[k] = (pretrainedmodels.inceptionv4(pretrained=None), stat['sparisity'])
        elif stat['net'] == 'resnet18':
            out[k] = (pretrainedmodels.resnet18(pretrained=None), stat['sparisity'])
        else:
            raise "Unknown net"
    return out

# ## OUTPUT YAML
# models_class:
#     modela: 1
#     modelb: 1
#     modelc: 2
#     modeld: 3
# cluster_center:
#     1: modela
#     2: modelc
#     3: modeld
def write_result(result):
    out = {
        "models_class": {},
        "cluster_center": {}
    }
    out["models_class"] = result
    for k, v in result.items():
        out["cluster_center"][v] = k
    import pdb; pdb.set_trace()
    with open("cluster_result.yaml", 'w') as f:
        yaml.dump(out, f)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python %s conf_file" % sys.argv[0])
        exit(1)
    conf_file = sys.argv[1]

    models = load_models(conf_file)
    features = gen_features(models)
    result = clustering(features)
    write_result(result)
