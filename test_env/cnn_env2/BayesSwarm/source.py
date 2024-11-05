#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/19/2020 """

import numpy as np

class Source:
    def __init__(self, id):
        self.id = id
        self.source_dim = 2
        if self.id == 0:
            self.source_0_init()
        elif self.id == 1:
            self.source_1_init()
        elif self.id == 2:
            self.source_2_init()
        elif self.id == 3:
            self.source_3_init()
        elif self.id == 4:
            self.source_4_init()
        elif self.id == 5:
            self.source_5_init()
        elif self.id == 51:
            self.source_51_init()
        elif self.id == 6:
            self.source_6_init()
        elif self.id == 7:
            self.source_7_init()
        elif self.id == 8:
            self.source_8_init()
        elif self.id == 9:
            self.source_9_init()
        elif self.id == 10:
            self.source_10_init()
        else:
            self.source_1_init()
        
    def get_source_location(self):
        
        return self.source_location

    def measure(self, location):
        if self.id == 0:
            signal_value = self.source_0_measure(location)
        elif self.id == 1:
            signal_value = self.source_1_measure(location)
        elif self.id == 2:
            signal_value = self.source_2_measure(location)
        elif self.id == 3:
            signal_value = self.source_3_measure(location)
        elif self.id == 4:
            signal_value = self.source_4_measure(location)
        elif self.id == 5:
            signal_value = self.source_5_measure(location)
        elif self.id == 51:
            signal_value = self.source_5_measure(location)
        elif self.id == 6:
            signal_value = self.source_6_measure(location)
        elif self.id == 7:
            signal_value = self.source_7_measure(location)
        elif self.id == 8:
            signal_value = self.source_8_measure(location)
        elif self.id == 9:
            signal_value = self.source_9_measure(location)
        elif self.id == 10:
            signal_value = self.source_10_measure(location)
        else:
            signal_value = self.source_1_measure(location)
        
        return signal_value


    def source_4_init(self):  # Based on Case 2 in MRS paper
        self.source_location = np.array([21, -19])
        self.time_max = 1000
        self.angular_range = np.array([0, 2 * np.pi])
        self.arena_lb = np.array([-24, -24])
        self.arena_ub = np.array([24, 24])
        self.source_detection_range = 0.2
        self.velocity = 0.2  # [m/s]
        self.decision_horizon_init = 50
        self.decision_horizon = 50
        self.local_penalizing_coef = {"M": 1.2, "L": 100}
        self.communication_range = 20

        # self.get_data_for_plot()

    def source_4_measure(self, location):
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

    def get_source_info(self):
        
        return self.velocity, self.decision_horizon, self.source_detection_range, self.source_location,\
                self.angular_range, self.time_max, self.arena_lb, self.arena_ub
    
    def get_source_info_arena(self):
        
        return self.angular_range, self.arena_lb, self.arena_ub
    
    def get_source_info_robot(self):
        
        return self.velocity, self.decision_horizon, self.decision_horizon_init, self.source_detection_range

    def get_source_info_mission(self):
        
        return self.source_location, self.time_max

    def get_source_bayes_settings(self):
        
        return self.local_penalizing_coef

    def set_velocity(self, velocity):
        Warning('Default velocity changed from {} to {}'.format(self.velocity, velocity))
        self.velocity = velocity
    
    def set_decision_horizon(self, decision_horizon):
        Warning('Default decision-horizon changed from {} to {}'.format(self.decision_horizon, decision_horizon))
        self.decision_horizon = decision_horizon
    
    def get_source_communication_range(self):
        return self.communication_range

    def get_data_for_plot(self):
        N = 100
        x1 = np.linspace(self.arena_lb[0], self.arena_ub[0], N)
        x2 = np.linspace(self.arena_lb[1], self.arena_ub[1], N)
        X1, X2 = np.meshgrid(x1,x2)
        X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
        Y = self.measure(X)
        Y = Y.reshape(N,-1)

        return X1, X2, Y