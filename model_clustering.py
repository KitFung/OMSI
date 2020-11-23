import os
import sys

import torch

# Do the clustering on torch model
# man craft feature: [sparity, number of fullcon, number of dropout, number of conv, number of activate, number of normalization, depth]
def gen_feature(model):
    pass

def gen_features(models):
    pass

# models: {
#   model_a: feature vector,
#   model_b: feature vector,
#   model_c: feature vector,
#   model_d: feature vector,
# }
def clustering(features):
    pass

# ## INPUT YAML
# config
#  - model_name_a: ./path/to/model_a
#  - model_name_b: ./path/to/model_b
#  - model_name_c: ./path/to/model_c
#  - model_name_d: ./path/to/model_d
def load_models(conf_file):
    pass

def write_result(result):
    pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python %s conf_file" % sys.argv[0])
        exit(1)
    conf_file = sys.argv[1]

    models = load_models(conf_file)
    features = gen_features(models)
    result = clustering(features)
    write_result(result)