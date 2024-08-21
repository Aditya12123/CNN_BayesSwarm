import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import scipy.stats as stats

path = r'F:\\ADAMS_Lab\\RAL_results\\Env_2\\freq_5'

complete_folder = r"\\complete\\mission_time_"
cnn_folder = r"\\cnn\\mission_time_"
src_folder = r"\\src\\mission_time_"
kmeans_folder = r"\\kmeans\\mission_time_"
ransac_folder = r"\\ransac\\mission_time_"

mission = "820.csv"

cnn_data = np.array(pd.read_csv(path + cnn_folder + mission))
src_data = np.array(pd.read_csv(path + src_folder+ mission))
kmeans_data = np.array(pd.read_csv(path + kmeans_folder + mission))
# ransac_data = np.array(pd.read_csv(path + ransac_folder + mission))
# complete_data = np.array(pd.read_csv(path + complete_folder))

t_stat, p_value = stats.ttest_ind(cnn_data, src_data)
print(p_value)
