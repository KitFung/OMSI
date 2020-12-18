import datetime
from collections import defaultdict
import pandas as pd
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

    def export_json(self, log_path):
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
        with open((log_path / 'profiler_dump.json'), 'w') as fp:
            json.dump(profiler_result, fp)

    def save_png(self, save_path, models_name_lookup):
        # this is messy because it is extracted from jupyter notebook
        pd.options.plotting.backend = "plotly"

        MEAN_WINDOW_SIZE = 200
        FIG_WIDTH = 1000
        FIG_HEIGHT = 600

        ENABLE_SAVE_IMAGE = True


        file1 = open( str( save_path / "models_name_lookup.txt"), "w")
        i=0
        for key, val in models_name_lookup.items():
            file1.write(key + ' : ' + str(i) + "\n")
            i+=1
        file1.close()


        # Load data from json
        data = json.load(open( str(save_path / 'profiler_dump.json')))


        # remove invalid filename characters
        def slugify(value):
            return "".join(x for x in value if x.isalnum())

        # ## Mean Accuracy Plot
        accumalted_scores = []
        selected_record = data['selected_record']
        for i in range(len(selected_record)):
            accumalted_scores.append(selected_record[i][-2])
        df = pd.DataFrame(accumalted_scores, index=range(len(accumalted_scores)), columns=['y'])
        df.index.name = 'Time'
        y_name = 'Mean accuracy (%)'
        df[y_name] = df.rolling(window=MEAN_WINDOW_SIZE).mean()
        fig = df.reset_index().plot.line(x='Time', y=y_name, width=FIG_WIDTH, height=FIG_HEIGHT)
        if ENABLE_SAVE_IMAGE:
            fig.write_image(str(save_path / (slugify(y_name) + ".png")))
        else:
            fig.show()


        load_model_ms = data['load_model_ms']
        iter_ms = data['iter_ms']

        contol_elapsed_over_t = []
        model_over_t = []

        for i in range(len(selected_record)):
            inference_ms = selected_record[i][-1]
            contol_elapsed_over_t.append(iter_ms[i] - load_model_ms[i] - inference_ms)
            #model_over_t.append(models_name_lookup[selected_record[i][0]])

        fps_over_t = []
        for i in range(len(iter_ms)):
            fps_over_t.append(1000 / iter_ms[i])

        # ## Mean Contol elapsed Time Plot
        y_name = 'Mean Control Elapsed Time (ms)'
        df = pd.DataFrame(contol_elapsed_over_t, index=range(len(contol_elapsed_over_t)), columns=['y'])
        df.index.name = 'Time'
        df[y_name] = df.rolling(window=MEAN_WINDOW_SIZE).mean()
        fig = df.reset_index().plot.line(x='Time', y=y_name, width=FIG_WIDTH, height=FIG_HEIGHT)
        # fig.update_yaxes(range=[0, 3])
        if ENABLE_SAVE_IMAGE:
            fig.write_image(str(save_path / (slugify(y_name) + ".png")))
        else:
            fig.show()

        # ## Mean FPS Plot
        y_name = 'Mean FPS'
        df = pd.DataFrame(fps_over_t, index=range(len(fps_over_t)), columns=['y'])
        df.index.name = 'Time'
        df[y_name] = df.rolling(window=MEAN_WINDOW_SIZE).mean()
        fig = df.reset_index().plot.line(x='Time', y=y_name, width=FIG_WIDTH, height=FIG_HEIGHT)
        if ENABLE_SAVE_IMAGE:
            fig.write_image(str(save_path / (slugify(y_name) + ".png")))
        else:
            fig.show()

        # ## Load Model elapsed time (ms) Plot
        y_name = 'Load Model elapsed time (ms)'
        df = pd.DataFrame(load_model_ms, index=range(len(load_model_ms)), columns=['y'])
        df.index.name = 'Time'
        fig = df.reset_index().plot.line(x='Time', y='y', width=FIG_WIDTH, height=FIG_HEIGHT)
        if ENABLE_SAVE_IMAGE:
            fig.write_image(str(save_path / (slugify(y_name) + ".png")))
        else:
            fig.show()


        y_name = 'Selected Model'
        df = pd.DataFrame(model_over_t, index=range(len(model_over_t)), columns=['y'])
        df.index.name = 'Time'
        # df[y_name] = df.rolling(window=1).mean()
        fig = df.reset_index().plot.scatter(x='Time', y='y', width=FIG_WIDTH, height=FIG_HEIGHT)
        # fig.update_yaxes(range=[0, 80])
        if ENABLE_SAVE_IMAGE:
            fig.write_image(str(save_path / (slugify(y_name) + ".png")))
        else:
            fig.show()
