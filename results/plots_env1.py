import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
os.environ['MPLCONFIGDIR'] = '/data1/users/abhatt4'
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"


# frequency 1
# path = r'F:\\ADAMS_Lab\\RAL_results\\Env_1\\freq_5' # Windows_path
path = "/data1/users/abhatt4/cnn_bayesswarm/CNN_BayesSwarm_RAL/RAL_results/Env_3/freq_1"

# Windows Paths
# folder_paths = [
#                 # r"\\complete\\mission_time_",
#                 r"\\cnn\\mission_time_",
#                 r"\\src\\mission_time_",
#                 r"\\kmeans\\mission_time_",
#                 r"\\ransac\\mission_time_",
# ]

# Lab machine path
folder_paths = [
                # "/complete/mission_time_",
                "/cnn/mission_time_",
                "/src/mission_time_",
                "/kmeans/mission_time_",
                "/ransac/mission_time_",
]


mission = '850.csv'
# Windows Paths
# env_path = r"env1_results\\freq_5\\"
# file_name = r"env1_50_robots_speed1.pdf"

#Lab machine 
env_path = "env3_results/"#freq_5/"#without_ransac/"#freq_5/"
file_name = "env3_50_robots_speed1.png"
title = "50 Robots"

fig, axs = plt.subplots(figsize=(6, 5.5))

final_array = np.array([])
data_list = []
for i, folder in enumerate(folder_paths):
    file_path = path + folder + mission
    file = pd.read_csv(file_path)
    df = pd.read_csv(file_path, header=None)  # Assuming no header
    arr = np.array(df)
    # arr = np.delete(arr, np.argmax(arr))
    # arr = np.delete(arr, np.argmax(arr))
    df = pd.DataFrame(arr)
    # windows
    # df['Category'] = folder.split('\\')[2]

    # Lab machine
    df['Category'] = folder.split('/')[1] 
    data_list.append(df)

ax = plt.subplot()
data = pd.concat(data_list, axis=0)
data.columns = ['Value', 'Category']
my_pal = {"ransac": "#FF80FF", "cnn": "orange", "src": "#b9e9e9", "kmeans": "#ffe1bd", 'kmeans_all': '#ffe1bd',
          "complete": "#b5651d", "max_signal": "purple"}
sns.boxplot(x='Category', y='Value', data=data, ax=ax, linewidth=1.5, linecolor='black',  palette=my_pal)  # S
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(["CNN", "SRC", "K-Means", "RANSAC"], fontsize=22, rotation=45)#, fontweight='bold')
ax.set_xlabel("")
# ax.set_yticks(np.arange(140, 250, 40)) # For Env-1 speed 0.2
# ax.set_yticks(np.arange(28, 44, 5)) # For Env-1 speed 1
ax.set_yticks(np.arange(40, 65, 10)) # For Env-2
# ax.set_yticks(np.arange(50, 250, 50)) # For Env-3
ax.set_ylabel("")
ax.set_ylabel("Mission Time [s]", fontsize=24)#, fontweight='bold')

    
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(30)
    tick.label2.set_fontsize(30)

    # tick.label1.set_fontproperties("monospace")
    # tick.label2.set_fontproperties("monospace")
    
plt.title(title, fontsize=24)
plt.tight_layout()
# Windows path
# save_image_path = r"F:\\ADAMS_Lab\\RAL_results\\results_plots\\" + env_path

# Lab machine path
save_image_path = "/data1/users/abhatt4/cnn_bayesswarm/CNN_BayesSwarm_RAL/RAL_results/results_plots/png_plots/" + env_path
plt.savefig(save_image_path + file_name, format='png', dpi=300, pad_inches=1)
plt.show()

