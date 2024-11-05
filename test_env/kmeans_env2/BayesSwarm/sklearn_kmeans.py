import os
from sklearn.cluster import KMeans
import numpy as np


def downsample(X, y, activate_set_size):
    obs = np.hstack((X, y.reshape(y.shape[0], 1)))
    kmeans = KMeans(n_clusters=activate_set_size, n_init='auto').fit(obs)
    down_obs = kmeans.cluster_centers_
    # os.system("pause")
    Xe = down_obs[:, :2]
    ye = down_obs[:, 2]

    return Xe, ye
