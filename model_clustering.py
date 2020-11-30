import os
import sys
import yaml

from sklearn.cluster import KMeans
import torch
from ptflops import get_model_complexity_info

input_shape = (3, 244, 244)

# Do the clustering on torch model
# # man craft feature: [sparity, number of fullcon, number of dropout, number of conv, number of activate, number of normalization, depth]
# man craft feature: [flops, macs, memory, sparity]
def gen_feature(model):
    macs, params = get_model_complexity_info(net, input_shape, as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    return params
# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/Swall0w/torchstat
# https://github.com/sovrasov/flops-counter.pytorch

def gen_features(models):
    return {name: gen_feature(model) for name, model in models.items()}

# models: {
#   model_a: feature vector,
#   model_b: feature vector,
#   model_c: feature vector,
#   model_d: feature vector,
# }
def clustering(features):
    names = features.keys()
    X = [features[k] for k in names]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    kmeans.labels_
    pass

# ## INPUT YAML
# config
#  - model_name_a: ./path/to/model_a
#  - model_name_b: ./path/to/model_b
#  - model_name_c: ./path/to/model_c
#  - model_name_d: ./path/to/model_d
def load_models(conf_file):
    with open(conf_file) as f:
        doc = yaml.full_load(file)
        return doc['config']

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
    for k, v in result:
        out["cluster_center"][v] = k

    with open("cluster.yaml", 'w') as f:
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