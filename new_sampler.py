import os
import random

import numpy as np
import matplotlib.pyplot as plt
from robot_source import Source


class Sampler:
    def __init__(self, resolution, obs_frequency, velocity):
        self.observation_frequency = obs_frequency
        self.velocity = velocity
        self.resolution = resolution
        self.robot_obs = {}
        self.iterations = 0
        self.new_array = []

    def reset(self, source, reset_counter):
        self.robot_iterations = 0
        self.data, self.lb, self.ub = source.get_info()
        self.source = source
        centre_pt = int(self.data.shape[0] / 2) + int(self.resolution / 2)
        self.centre_pt = centre_pt
        self.dist_x = self.dist_y = self.observation_frequency * self.velocity

        r = 100
        store_arr = []

        robots_arr = [20, 30, 50]
        distance_arr = [10, 15, 22, 30, 50]

        n_robots_counter = np.int32(reset_counter/len(distance_arr))
        n_robots_counter = np.mod(n_robots_counter, len(robots_arr))
        dist_tr_counter = np.int32(reset_counter / 1)
        dist_tr_counter = np.mod(dist_tr_counter, len(distance_arr))

        self.n_robots = robots_arr[n_robots_counter]
        self.x = 0

        self.distance_tr = distance_arr[dist_tr_counter]
        q = 1
        self.q = q

        print(f"Robots: {self.n_robots}, dist: {self.distance_tr}, angle: {q}")

        pts = self.data[centre_pt, :2]

        # for q in range(10):
        for k in range(self.n_robots):
            theta = 1 * (k + q) * 360 / self.n_robots
            theta = theta * np.pi / 180

            for j in range(self.distance_tr):
                x = self.data[centre_pt, 0] + np.cos(theta) * j * self.dist_x
                y = self.data[centre_pt, 1] + np.sin(theta) * j * self.dist_y
                pts = np.vstack((pts, np.array([x, y])))

            store_arr.append(pts[len(pts) - 1])
            if len(pts) >= r:
                break

            pts = np.clip(pts, self.lb, self.ub)

        while len(pts) < r:
            pts, store_arr = self.random_observations(pts, store_arr)
        D_old = pts
        self.iterations = 0
        self.new_array = []

        return D_old, store_arr

    def create_new_obs(self, D_old, store_arr, just_random_obs):

        if just_random_obs == 0:
            new_obs, store_arr_new = self.random_observations(D_old, store_arr)
            return new_obs, store_arr_new

        if len(store_arr) < self.n_robots:
            new_obs, store_arr_new = self.remaining_lines(D_old, store_arr)
            return new_obs, store_arr_new

        elif just_random_obs == 1:
            # print("CEntre Lines")
            # os.system('pause')
            new_obs, store_arr_new = self.random_centre_lines(D_old, store_arr)
            return new_obs, store_arr_new

        else:
            new_obs, store_arr_new = self.random_observations(D_old, store_arr)
            return new_obs, store_arr_new

    def remaining_lines(self, D_old, store_arr):
        store_arr_new = store_arr.copy()
        centre_pt = self.centre_pt
        theta = (len(store_arr) + self.q) * 360 / self.n_robots
        theta = theta * np.pi / 180
        pts = D_old[len(D_old) - 1, :2].reshape(1, 2)

        for j in range(self.distance_tr):
            x = self.data[centre_pt, 0] + np.cos(theta) * j * self.dist_x
            y = self.data[centre_pt, 1] + np.sin(theta) * j * self.dist_y
            # print(np.array([x, y]), pts)
            # os.system("pause")
            pts = np.vstack((pts, np.array([x, y])))

        store_arr_new.append(pts[len(pts) - 1])
        new_obs = np.vstack((D_old[:, :2], pts))

        # plt.scatter(new_obs[:, 0], new_obs[:, 1]), plt.title("Remaining_Lines")
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return np.clip(new_obs, self.lb, self.ub), store_arr_new

    def random_centre_lines(self, D_old, store_arr):

        store_arr_new = store_arr.copy()
        centre_pt = self.centre_pt
        theta = np.random.randint(1, 359)
        theta = theta * np.pi / 180
        pts = D_old[len(D_old) - 1, :2].reshape(1, 2)

        for j in range(self.distance_tr):
            x = self.data[centre_pt, 0] + np.cos(theta) * j * self.dist_x
            y = self.data[centre_pt, 1] + np.sin(theta) * j * self.dist_y
            # print(np.array([x, y]), pts)
            # os.system("pause")
            pts = np.vstack((pts, np.array([x, y])))

        store_arr_new.append(pts[len(pts) - 1])
        new_obs = np.vstack((D_old[:, :2], pts))

        # plt.scatter(new_obs[:, 0], new_obs[:, 1]), plt.title("Remaining_Lines")
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return np.clip(new_obs, self.lb, self.ub), store_arr_new

    def random_observations(self, D_old, store_arr):
        # print(len(store_arr))
        # os.system('pause')
        store_arr_new = store_arr.copy()
        current_index = np.mod(self.iterations, len(store_arr))
        # current_index = np.random.randint(0, len(store_arr))
        theta = np.random.randint(10, 350)
        theta = theta * np.pi / 180
        distance_tr = np.random.randint(3, 50)
        req_pts = store_arr_new[current_index]
        pts = D_old[len(D_old) - 1, :2].reshape(1, 2)

        for j in range(distance_tr):
            x = req_pts[0] + np.cos(theta) * j * self.dist_x
            y = req_pts[1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        store_arr_new[current_index] = pts[len(pts) - 1]

        new_obs = np.vstack((D_old[:, :2], pts))
        self.iterations += 1
        # plt.scatter(new_obs[:, 0], new_obs[:, 1])
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return np.clip(new_obs, self.lb, self.ub), store_arr_new

    def initial_shape_reset(self):
        centre_pt = self.centre_pt
        pts = self.data[centre_pt, :2]
        r = 100
        store_arr = []
        q = np.random.rand()

        for k in range(self.n_robots):
            theta = 1 * (k + q) * 360 / self.n_robots
            theta = theta * np.pi / 180

            for j in range(self.distance_tr):
                x = self.data[centre_pt, 0] + np.cos(theta) * j * self.dist_x
                y = self.data[centre_pt, 1] + np.sin(theta) * j * self.dist_y
                pts = np.vstack((pts, np.array([x, y])))

            store_arr.append(pts[len(pts) - 1])
            if len(pts) >= r:
                break

            pts = np.clip(pts, self.lb, self.ub)

        while len(pts) < r:
            pts, store_arr = self.random_observations(pts, store_arr)
        # D_old = self.source.measure_signal(pts)
        # D_old = pts
        self.iterations = 0
        self.new_array = []
        # plt.scatter(D_old[:, 0], D_old[:, 1]), plt.title(f"total_obs: {len(D_old)}, r: {r}, n_robots: {self.n_robots}")
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return pts, store_arr

    def just_one_robot(self, D_old, store_arr):
        # print(len(store_arr))
        # os.system('pause')
        store_arr_new = store_arr.copy()
        if self.x < len(store_arr):
            if self.robot_iterations % 15 == 0:
                self.x += np.random.randint(1, 3)
        else:
            self.x = 1
        current_index = np.mod(self.x, len(store_arr))

        # current_index = np.random.randint(0, len(store_arr))
        theta = np.random.randint(10, 350)
        theta = theta * np.pi / 180
        distance_tr = np.random.randint(3, 10)
        req_pts = store_arr_new[current_index]
        pts = D_old[len(D_old) - 1, :2].reshape(1, 2)

        for j in range(distance_tr):
            x = req_pts[0] + np.cos(theta) * j * self.dist_x
            y = req_pts[1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        store_arr_new[current_index] = pts[len(pts) - 1]

        new_obs = np.vstack((D_old[:, :2], pts))
        self.robot_iterations += 1
        # self.iterations += 1
        # plt.scatter(new_obs[:, 0], new_obs[:, 1])
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return np.clip(new_obs, self.lb, self.ub), store_arr_new