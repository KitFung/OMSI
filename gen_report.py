import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

profile_fp = 'profiler_dump.json'
if len(sys.argv) > 1:
    profile_fp = sys.argv[1]

f = open(profile_fp)
res = json.load(f)


window_size = 200
mean_accuracy_over_t = []
score = []


lookup = {
    'mobilenet_34.67_66.84': 0,
    'mobilenet_36.04_37.97': 1,
    'mobilenet_41.75_33.35': 2,
    'resnet18_39.52_66.54': 3,
    'resnet18_40.87_46.49': 4,
    'resnet18_42.63_15.38': 5,
    'resnet18_58.17_71.87': 6,
    'resnet18_58.66_55.21': 7,
}

selected_record = res['selected_record']


for i in range(len(selected_record)):
    score.append(selected_record[i][1])
    if i < window_size:
        mean_accuracy_over_t.append(None)
    else:
        mean_accuracy_over_t.append(np.mean(score[-1*window_size:]))

load_model_ms = res['load_model_ms']
iter_ms = res['iter_ms']

contol_elapsed_over_t = []
model_over_t = []

for i in range(len(selected_record)):
    inference_ms = selected_record[i][2]
    contol_elapsed_over_t.append(iter_ms[i] - load_model_ms[i] - inference_ms)
    model_over_t.append(lookup[selected_record[i][0]])

fps_over_t = []
for i in range(len(iter_ms)):
    fps_over_t.append(1000/iter_ms[i])

print('logic_elapsed_over_t', len(contol_elapsed_over_t))
print('mean_accuracy_over_t', len(mean_accuracy_over_t))

'''
'model_score'
'selected_record'
'model_pull_cnt'
'model_ms'
'total_ms'
'total_score'
'iter_ms'
'load_model_ms'
'''

start_t = time.perf_counter()
time.sleep(1)
end_t = time.perf_counter()
# print(end_t - start_t)
df = pd.DataFrame(contol_elapsed_over_t, columns=['contol_elapsed_over_t'])
ax = df.plot(y='contol_elapsed_over_t', figsize=(50, 5), xticks=range(len(df)), marker=',', linestyle=None)
plt.savefig("contol_elapsed_over_t.png")
df.to_csv('contol_elapsed_over_t.csv')

df = pd.DataFrame(mean_accuracy_over_t, columns=['mean_accuracy_over_t'])
ax = df.plot(y='mean_accuracy_over_t', figsize=(50, 5), xticks=range(len(df)), marker=',', linestyle=None)
plt.savefig("mean_accuracy_over_t.png")
df.to_csv('mean_accuracy_over_t.csv')


df = pd.DataFrame(fps_over_t, columns=['fps_over_t'])
ax = df.plot(y='fps_over_t', figsize=(50, 5), xticks=range(len(df)), marker=',', linestyle=None)
plt.savefig("fps_over_t.png")
df.to_csv('fps_over_t.csv')


df = pd.DataFrame(model_over_t, columns=['model_over_t'])
ax = df.plot(y='model_over_t', figsize=(50, 5), xticks=range(len(df)), marker=',', style='o')
plt.savefig("model_over_t.png")
df.to_csv('model_over_t.csv')

'''
================    ===============================
character           description
================    ===============================
   -                solid line style
   --               dashed line style
   -.               dash-dot line style
   :                dotted line style
   .                point marker
   ,                pixel marker
   o                circle marker
   v                triangle_down marker
   ^                triangle_up marker
   <                triangle_left marker
   >                triangle_right marker
   1                tri_down marker
   2                tri_up marker
   3                tri_left marker
   4                tri_right marker
   s                square marker
   p                pentagon marker
   *                star marker
   h                hexagon1 marker
   H                hexagon2 marker
   +                plus marker
   x                x marker
   D                diamond marker
   d                thin_diamond marker
   |                vline marker
   _                hline marker
================    ===============================
'''