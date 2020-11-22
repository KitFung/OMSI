class OMSILoader(object):
    def __init__(self):
        pass

    def load(self, root_path):
        pass

    def model_store(self):
        pass

    # return list of tuple (model_name, distance to target model)
    # only include model in same cluster
    def model_distance(self, target_model):
        pass

    # return the expected accuracy of the model
    # if the distribution is given, return the weighted distribution
    def expected_accuracy(self, target_model, label_distribution=None):
        pass