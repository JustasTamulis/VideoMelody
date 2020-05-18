import glob
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut
from pypianoroll import Multitrack, Track
import math
import seaborn as sns
import random
# Metrics calculator built on https://github.com/RichardYang40148/mgeval.
# Using demo and postprint as reference.

MODELS = [0,1,4]
midi_path = "C:\\VideoMelody\\results\\"
num_samples = 140
metrics_list = ['total_used_pitch',
                'bar_used_pitch',
                'total_used_note', 'bar_used_note',
                'total_pitch_class_histogram',
                'bar_pitch_class_histogram',
                'pitch_class_transition_matrix',
                'pitch_range',"avg_pitch_shift",
                'avg_IOI',
                'note_length_hist',
                'note_length_transition_matrix'
                ]

eval = [{},{},{},{},{}]
intra = [None] *5
inter = [None] *5
plot_intra = [None] *5
plot_inter = [None] *5
set = [0,0,0,0,0]
for i in range(0,5):
    for metric in metrics_list:
        if metric == 'total_pitch_class_histogram':
            eval[i][metric] = np.zeros((num_samples,12))
        elif metric == 'bar_pitch_class_histogram':
            eval[i][metric] = np.zeros((num_samples,24,12))
        elif metric == 'pitch_class_transition_matrix':
            eval[i][metric] = np.zeros((num_samples,12,12))
        elif metric == 'note_length_hist':
            eval[i][metric] = np.zeros((num_samples,12))
        elif metric == 'note_length_transition_matrix':
            eval[i][metric] = np.zeros((num_samples,12,12))
        else:
            eval[i][metric] = np.zeros((num_samples,1))
    set[i] = glob.glob(midi_path  + str(i) + '\\*.mid')
    intra[i] = np.zeros((num_samples, len(metrics_list), num_samples-1))
    inter[i] = np.zeros((num_samples, len(metrics_list), num_samples))
corrupted = list()
for metric in metrics_list:
    print(metric)
    for model in MODELS:
        for i in range(0, num_samples):
            feature = core.extract_feature(set[model][i])
            attr = getattr(core.metrics(),metric)(feature)
            # Check if output is not NaN
            if isinstance(attr, float) or isinstance(attr, int):
                if math.isnan(attr):
                    corrupted.append(i)
            else:
                for gg in attr.flatten():
                    if math.isnan(gg):
                        corrupted.append(i)
            eval[model][metric][i] = attr
corrupted = np.unique(corrupted)
print(corrupted)
# Delete corrupted samples
for model in MODELS:
    for metric in metrics_list:
        eval[model][metric] = np.delete(eval[model][metric],corrupted,0)
num_samples = num_samples - len(corrupted)

for i in range(0,5):
    set[i] = glob.glob(midi_path  + str(i) + '\\*.mid')
    intra[i] = np.zeros((num_samples, len(metrics_list), num_samples-1))
    inter[i] = np.zeros((num_samples, len(metrics_list), num_samples))

loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
for i in range(len(metrics_list)):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        for m in MODELS:
            intra[m][test_index[0]][i] = utils.c_dist(eval[m][metrics_list[i]][test_index], eval[m][metrics_list[i]][train_index])

# print(eval[0]['total_pitch_class_histogram'])

for i in range(len(metrics_list)):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        for m in MODELS:
            inter[m][test_index[0]][i] = utils.c_dist(eval[m][metrics_list[i]][test_index], eval[0][metrics_list[i]])


for m in MODELS:
    plot_intra[m] = np.transpose(intra[m],(1, 0, 2)).reshape(len(metrics_list), -1)
    plot_inter[m] = np.transpose(inter[m],(1, 0, 2)).reshape(len(metrics_list), -1)

# print(plot_intra[0])
#
# print(len(plot_intra))
# print(plot_intra[0].shape)
# print(len(intra))
# print(intra[0].shape)

for i in range(len(metrics_list)):
    s = ''
    metric = metrics_list[i]
    print("----------" + metric + "--------------")
    # print("mean " + str(round(np.mean(plot_intra[0][i]),2)))
    # print(" std " + str(round(np.std(plot_intra[0][i]),2)))
    s = str(round(np.mean(plot_intra[0][i]),2)) + '  &  ' + str(round(np.std(plot_intra[0][i]),2)) + '  &  '

    mn = 0
    std =0
    kld = 0
    oa = 0

# Calculate KLD and OA for each model / metric.
    for m in [m for m in MODELS if m != 0]:
        print("....." + str(m) + "....")
        print("mean " + str(round(np.mean(plot_intra[m][i]),2)))
        print("std  " + str(round(np.std(plot_intra[m][i]),2)))

        print("KLD " + str(round(utils.kl_dist(plot_intra[m][i], plot_inter[m][i]),2)))
        print("OA  " + str(round(utils.overlap_area(plot_intra[m][i], plot_inter[m][i]),2)))
        # s = s + str(round(np.mean(plot_intra[m][i]),2)) + '  &  ' + str(round(np.std(plot_intra[m][i]),2)) + '  &  '
        # s = s  + str(round(utils.kl_dist(plot_intra[m][i], plot_inter[m][i]),2)) + '  &  ' + str(round(utils.overlap_area(plot_intra[m][i], plot_inter[m][i]),2)) + '  &  '
    # print(s)


linestyles = ["-", "--", "xx", 'dashdot', 'dashdot']

# Plot PDFs for each metric
for i in range(len(metrics_list)):

    sns.kdeplot(plot_intra[0][i], label='train_intra', shade=True, color="r")
    for m in [m for m in MODELS if m != 0]:
        sns.kdeplot(plot_inter[m][i], label='Model ' + str(m) + ' inter', linestyle=linestyles[m], color="b")
        sns.kdeplot(plot_intra[m][i], label='Model ' + str(m) +' intra', linestyle=linestyles[m], color="g")

    plt.title(metrics_list[i])
    plt.xlabel('Euclidean distance')

    plt.ylabel('Density')
    plt.show()
