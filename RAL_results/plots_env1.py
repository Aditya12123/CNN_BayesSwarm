import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"


# frequency 1
path = r'F:\\ADAMS_Lab\\RAL_results\\Env_1\\freq_5'

folder_paths = [
                # r"\\complete\\mission_time_",
                r"\\cnn\\mission_time_",
                r"\\src\\mission_time_",
                r"\\kmeans\\mission_time_",
                r"\\ransac\\mission_time_",
]

mission = '850.csv'
env_path = r"env1_results\\freq_5\\"
file_name = r"env1_50_robots_speed1.pdf"

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
    df['Category'] = folder.split('\\')[2]
    data_list.append(df)

ax = plt.subplot()
data = pd.concat(data_list, axis=0)
data.columns = ['Value', 'Category']
my_pal = {"ransac": "#FF80FF", "cnn": "orange", "src": "#b9e9e9", "kmeans": "#ffe1bd", 'kmeans_all': '#ffe1bd',
          "complete": "#b5651d", "max_signal": "purple"}
sns.boxplot(x='Category', y='Value', data=data, ax=ax, linewidth=1.5, linecolor='black',  palette=my_pal)  # S
ax.set_xticklabels(["CNN", "SRC", "K-Means", "RANSAC"], fontsize=22, rotation=45)#, fontweight='bold')
ax.set_xlabel("")
ax.set_yticks(np.arange(28, 44, 5)) # For Env-1 speed 1
# ax.set_yticks(np.arange(140, 250, 40)) # For Env-1 speed 0.2
# ax.set_yticks(np.arange(40, 70, 10)) # For Env-2
# ax.set_yticks(np.arange(50, 400, 100)) # For Env-3
ax.set_ylabel("")
ax.set_ylabel("Mission Time [s]", fontsize=24)#, fontweight='bold')

    
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(30)
    tick.label2.set_fontsize(30)

    # tick.label1.set_fontproperties("monospace")
    # tick.label2.set_fontproperties("monospace")
    

plt.tight_layout()
save_image_path = r"F:\\ADAMS_Lab\\RAL_results\\results_plots\\" + env_path
plt.savefig(save_image_path + file_name, format='pdf', dpi=300, pad_inches=1)
plt.show()

