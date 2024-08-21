#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import numpy as np
import matplotlib.pyplot as plt

class Source:
    def __init__(self, id):
        self.id = id
        self.source_dim = 2
        self.source_8_init()
        
    def get_source_location(self):
        
        return self.source_location

    def measure(self, location):

        signal_value = self.source_8_measure(location)
        
        return signal_value

    def source_8_init(self):  # Based on Case 2 in MRS paper
        self.source_location = np.array([21, -19])
        self.time_max = 1000
        self.angular_range = np.array([0, 2 * np.pi])
        self.arena_lb = np.array([-24, -24])
        self.arena_ub = np.array([24, 24])
        self.source_detection_range = 0.2
        self.velocity = 1  # [m/s]
        self.decision_horizon_init = 10
        self.decision_horizon = 10
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20

        # self.get_data_for_plot()

    def source_8_measure(self, location):
        c = self.source_location
        x = location
        sig1 = -300
        sig2 = -40
        coef = 0.4

        dx = x - c
        if np.size(location) > self.source_dim:
            dx2 = np.linalg.norm(dx, axis=1) ** 2
        else:
            dx2 = np.dot(dx, dx)
        f = np.exp(dx2 / sig1)
        c_list = np.array([[0, -15],\
                           [-19, 10], [21, 0], [-15, -15]])
        n, _ = c_list.shape

        for i in range(n):
            dx = x - c_list[i, :]
            if np.size(location) > self.source_dim:
                dx2 = np.linalg.norm(dx, axis=1) ** 2
            else:
                dx2 = np.dot(dx, dx)
            f += coef * np.exp(dx2 / sig2)

        f = np.where(f < 0, 0, f)

        return f    

    def get_data_for_plot(self):
        N = 100
        x1 = np.linspace(self.arena_lb[0], self.arena_ub[0], N)
        x2 = np.linspace(self.arena_lb[1], self.arena_ub[1], N)
        X1, X2 = np.meshgrid(x1,x2)
        X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
        Y = self.measure(X)
        Y = Y.reshape(N,-1)

        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.contour(X1, X2, Y)
        path = 'F:\ADAMS_Lab\RAL_results\Env_2\signal_contour'
        fig.savefig(path + 'env_2.pdf', format='pdf', dpi=300, bbox_inches = 'tight', pad_inches=0.5)
        plt.show()

        return X1, X2, Y
    


source = Source(id=8)
source.get_data_for_plot()
