import datetime
from collections import defaultdict

import numpy as np
import json


class Profiler(object):
    def __init__(self, window_size, required_fps, model_cluster):
        self.window_size = window_size
        self.model_pull_cnt = defaultdict(int)
        self.model_score = defaultdict(list)
        self.model_ms = defaultdict(list)
        self.total_score = defaultdict(float)
        self.total_ms = defaultdict(float)
        self.invalid = set()
        self.selected_record = []
        self.iter_ms = []
        self.load_model_ms = []
        self.model_cluster = model_cluster
        self.threshold_ms = 1000.0 / required_fps
        self.explore_threshold = 1000

    # Maybe consider both variance and mean
    def model_acceptable(self, name):
        if not self.explore_enough(name):
            return True
        return np.mean(self.model_ms[name][1:])

    def explore_enough(self, name):
        return self.model_pull_cnt[name] >= self.explore_threshold

    def cluster_rank(self):
        cluster_best_score = {}
        for model, tot_score in self.total_score.items():
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
        self.selected_record.append((name, score, ms))
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

    def log_single_iter(self, ms):
        self.iter_ms.append(ms)

    def log_load_model(self, ms):
        self.load_model_ms.append(ms)

    def export_json(self):
        # 1. The fps over time # iter_ms
        # 2. The acc over time # selected_record
        # 3. The pull arm over time
        # 4. The input distribute over time
        profiler_result = defaultdict()
        profiler_result['model_score'] = self.model_score
        profiler_result['selected_record'] = self.selected_record
        profiler_result['model_pull_cnt'] = self.model_pull_cnt
        profiler_result['model_ms'] = self.model_ms
        profiler_result['total_ms'] = self.total_ms
        profiler_result['total_score'] = self.total_score
        profiler_result['iter_ms'] = self.iter_ms
        profiler_result['load_model_ms'] = self.load_model_ms

        current_time = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        with open('profiler_dump_' + current_time + '.json', 'w') as fp:
            json.dump(profiler_result, fp)
