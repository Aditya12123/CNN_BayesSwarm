import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "Arial"


absolute_path = r"F:\\ADAMS_Lab\\CCR_Train\\cnn_train_env\\RAL_results\\scale_agnostic_bayesswarm"
original = r'\\BayesSwarm'
cnn = r'\\cnn'
file = r'\\robot_3_computing_time.csv'
studies = [ r'\\case_810', r'\\case_811', r'\\case_812']
mt_studies = [r'810.csv', r'811.csv', r'812.csv']
# file_dict = {}
colors = ['blue', 'red', 'green']
robots = ['Robots: 10', 'Robots: 20', 'Robots: 30']
rts = [10, 20, 30]

fig, ax1 = plt.subplots(figsize=(8, 7.5))
mission_time_CNN = []
mission_time_Original = []
for i, study in enumerate(studies):
    df = pd.read_csv(absolute_path + cnn + study + file, header=None)
    file_path_original = np.array(pd.read_csv(absolute_path + original + study + file, header=None))
    file_path_cnn = np.array(pd.read_csv(absolute_path + cnn + study + file, header=None))

    no_dec_or = file_path_original[:,2]
    no_dec_cnn = file_path_cnn[:, 2]

    ax1.plot(no_dec_or, file_path_original[:, 1], marker='X', color=colors[i], linestyle='solid')#, label=f'{robots[i]} - BayesSwarm')
    ax1.plot(no_dec_cnn, file_path_cnn[:, 1], marker='s', color=colors[i], linestyle='dashed')#, label=f'{robots[i]} - CNN-BayesSwarm')

    ax1.set_xticks(ticks=[1, 2, 3, 4, 5])

for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontsize(30)
    tick.label2.set_fontsize(30)

for tick in ax1.xaxis.get_major_ticks():
    tick.label1.set_fontsize(30)
    tick.label2.set_fontsize(30)

plt.xlabel('Number of Decisions', fontsize=24)
plt.ylabel('Computing Time per Decision [s]', fontsize=24)
plt.yticks(np.arange(0, 37, 8))

font = FontProperties()
font.set_size(18)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = list(set(labels))
labels_order = ['BayesSwarm', 'CNN-BayesSwarm']
plt.legend(labels_order, prop=font)
save_image_path = r"F:\\ADAMS_Lab\\CCR_Train\\cnn_train_env\\RAL_results\\results_plots\\scale_agnostic_bs" 
plt.savefig(save_image_path + r'\\scale_agnostic_bs.pdf', format='pdf', dpi=300, pad_inches=1)
plt.show()
