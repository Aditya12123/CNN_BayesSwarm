import os

import numpy as np
from numpy import arange, meshgrid, exp, cos, pi, sqrt, e
import matplotlib.pyplot as plt


class Source():
    def __init__(self, resolution):
        self.resolution = resolution
        self.signal = 1
        self.data = None
        self.ub = None
        self.lb = None

    def generate_arena(self, epoch):
        """To be run at every reset. This will generate data for the sampler."""
        ele = 300
        self.lb = [-ele, -ele]
        self.ub = [ele, ele]
        self.data = self.get_data(self.lb, self.ub, self.resolution)
    
    def get_data(self, lb = [-5, -5], ub = [5, 5], resolution=100, signal=1):
        """
        input:
        -lb: Lower bound for the env.
        -ub: Upper bound for the env.
        -resolution: Shape of the functional space (resolution, resolution) 

        output:
        -data: Array of shape (resolution**2, 3) containing locations X1, X2 and the signal measurements Y
        """
        x1 = np.linspace(lb[0], ub[0], resolution)
        x2 = np.linspace(lb[1], ub[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        # Y, self.a, self.b = self.unimodal_signal(X1, X2) if signal == 1 else self.multimodal_signal(X1, X2)
        data = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        # print("Data_ :", data[2][2])
        return data

    def get_info(self):
        return self.data, self.lb, self.ub