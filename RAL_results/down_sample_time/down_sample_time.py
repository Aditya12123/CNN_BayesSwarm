import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "Arial"

absolute_path = r"F:\\ADAMS_Lab\\CCR_Train\\cnn_train_env\\RAL_results\\down_sample_time\\int_fact_time_41.csv"

cnn = np.array(pd.read_csv("log_time_cnn_new.csv"))
ransac = np.array(pd.read_csv('log_time_ransac.csv'))
src = np.array(pd.read_csv('int_fact_time_41.csv'))[:100, 0]
kmeans_or = np.array(pd.read_csv('kmeans_time_41.csv'))[:100, 0]
min_length = min(len(cnn), len(ransac),  len(src), len(kmeans_or))

fig, _ = plt.subplots(figsize=(8.5, 8))

plt.plot(range(min_length), ransac[:min_length], color='r', label='RANSAC')
plt.plot(range(min_length), src[:min_length], color='orange', label='SRC', marker='X')
plt.plot(range(min_length), kmeans_or[:min_length], color='pink', label='K-Means')
plt.plot(range(min_length), cnn[:min_length], color='m', label='CNN')

# plt.title('Computation Time for Down-Sampling', fontweight='bold', fontsize=18)
plt.xlabel('Time Steps', fontsize=24), plt.ylabel('Computation Time[s]', fontsize=24)
plt.xticks(fontsize=30), plt.yticks(np.arange(0, 2, 0.4), fontsize=30)
font = FontProperties()
font.set_size(20)
plt.legend(prop=font)

save_image_path = r"F:\\ADAMS_Lab\\CCR_Train\\cnn_train_env\\RAL_results\\results_plots\\down_sample_time" 
plt.savefig(save_image_path + r'\\downsample_time_compare.pdf', format='pdf', dpi=300, pad_inches=1)
plt.show()
