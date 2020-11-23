from collections import defaultdict

import numpy as np

class Profiler(object):
    def __init__(self, window_size, required_fps, model_cluster):
        self.window_size = window_size
        self.model_pull_cnt = defaultdict(int)
        self.model_score = defaultdict(list)
        self.model_ms = {}
        self.total_score = {}
        self.total_ms = {}
        self.invalid = set()

        self.model_cluster = model_cluster
        self.threshold_ms = 1000.0 / required_fps
        self.explore_threshold = 20

    # Maybe consider both variance and mean
    def model_acceptable(self, name):
        if not self.explore_enough(name):
            return True
        return self.total_ms[name] / len(self.total_ms)

    def explore_enough(self, name):
        return self.model_pull_cnt[name] >= self.explore_threshold

    def cluster_rank(self):
        cluster_best_score = {}
        for model, tot_score in self.total_score:
            if model not in self.invalid:
                score = tot_score / len(self.model_score[model])
                cluster = self.model_cluster[model]
                if cluster_best_score.get(cluster, None) is None:
                    cluster_best_score[cluster] = score
                else:
                    if cluster_best_score[cluster] < score:
                        cluster_best_score[cluster] = score
        tlist = list(cluster_best_score.items())
        tlist.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in tlist]
    
    def invalid_model(self, name):
        self.invalid.add(name)

    def profile_once(self, name, score, ms):
        self.model_pull_cnt[name] += 1

        self.model_score[name].append(score)
        self.model_ms[name].append(ms)
        self.total_score[name] += score
        self.total_ms[name] += ms

        if len(self.model_score[name]) > self.window_size:
            tscore = self.model_score[name].pop(0)
            tms = self.model_ms[name].pop(0)
            self.total_score[name] -= tscore
            self.total_ms[name] -= tms


    def draw_report(self):
        # 1. The fps over time
        # 2. The acc over time
        # 3. The pull arm over time
        # 4. The input distribute over time
        pass
