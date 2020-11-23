import yaml
import numpy as np

from load_models import ModelStore

"""
Expected conf format:
models:
    modela: onnx_patha
    modelb: onnx_pathb
    modelc: onnx_pathc
    modeld: onnx_pathd
models_class:
    modela: 1
    modelb: 1
    modelc: 2
    modeld: 3
cluster_center:
    1: modela
    2: modelc
    3: modeld
expected_accuracy:
    modela: # array of size 10
    - 0.89
    - 0.99
    - 0.78
    ...
    - 0.39
    modelb: # array of size 10
    - 0.78
    ...
"""
class OMSILoader(object):
    def __init__(self):
        self.conf = None
        self.store = None
        self.acc = {}

    # root_conf should be a yaml
    def load(self, root_conf):
        with open(root_conf) as f:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            conf = yaml.load(f, Loader=yaml.FullLoader)
            self.conf = conf

        self.store = ModelStore(
            self.conf['models'],
            self.conf['models_class'],
            self.conf['cluster_center'])
        
        for k, v in self.conf['expected_accuracy']:
            self.acc[k] = np.array(v)

    def model_store(self):
        return self.store

    # return the expected accuracy of the model
    # if the distribution is given, return the weighted distribution
    def expected_accuracy(self, target_model, label_distribution=None):
        if label_distribution is None:
            return self.acc[target_model].mean()
        normalize_dis = np.array(label_distribution)
        normalize_dis /= normalize_dis.sum()
        score = self.acc[target_model] * normalize_dis
        return score.sum()


if __name__ == '__main__':
    loader = OMSILoader()
    loader.load('./omsi_conf.yaml')
    import pdb; pdb.set_trace()
    pass
