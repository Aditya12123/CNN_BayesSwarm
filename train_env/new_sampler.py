import os
import random

import numpy as np
import matplotlib.pyplot as plt


class Sampler:
    def __init__(self, resolution, obs_frequency, velocity):
        self.observation_frequency = obs_frequency
        self.velocity = velocity
        self.resolution = resolution
        self.iterations = 0
        self.robots_arr = [20, 30, 50]
        self.distance_arr = [10, 15, 22, 30, 50]

    def generate_initial_trajectory(self):
        """
        Generates Initial trajectory as seen in BayesSwarm

        """
        location_data = self.data[self.centre_pt, :2]
        trajectory_endpoints = []
        min_datapoints = 100

        for k in range(self.n_robots):
            theta = 1 * (k + 1) * 360 / self.n_robots
            theta = theta * np.pi / 180

            for j in range(self.trajectory_length):
                x = self.data[self.centre_pt, 0] + np.cos(theta) * j * self.dist_x
                y = self.data[self.centre_pt, 1] + np.sin(theta) * j * self.dist_y
                location_data = np.vstack((location_data, np.array([x, y])))

            trajectory_endpoints.append(location_data[len(location_data) - 1])
            if len(location_data) >= min_datapoints:
                break

            location_data = np.clip(location_data, self.lb, self.ub)

        while len(location_data) < min_datapoints:
            location_data, trajectory_endpoints = self.random_observations(location_data, trajectory_endpoints)
        
        return location_data, trajectory_endpoints

    def reset(self, source, reset_counter):
        """

        :param source:
        :param reset_counter:
        :return:
        """
        self.robot_iterations = 0
        self.data, self.lb, self.ub = source.get_info()
        self.centre_pt = int(self.data.shape[0] / 2) + int(self.resolution / 2)
        self.dist_x = self.dist_y = self.observation_frequency * self.velocity

        n_robots_counter = np.int32(reset_counter/len(self.robots_arr))
        n_robots_counter = np.mod(n_robots_counter, len(self.robots_arr))
        dist_tr_counter = np.int32(reset_counter / 1)
        dist_tr_counter = np.mod(dist_tr_counter, len(self.distance_arr))

        self.n_robots = self.robots_arr[n_robots_counter]
        self.x = 0

        self.trajectory_length = self.distance_arr[dist_tr_counter]

        print(f"Robots: {self.n_robots}, dist: {self.trajectory_length}")

        location_data, trajectory_endpoints = self.generate_initial_trajectory()

        self.iterations = 0

        return location_data, trajectory_endpoints

    def generate_new_trajectories(self, current_trajectories, trajectories_endpoint, new_trajectory_type):
        """

        :param current_trajectories: Data-set of all the locations visited by the robots (Trajectories generated so far)
        :param trajectories_endpoint: Last location data-point of all the trajectories generated so far
        :param new_trajectory_type:
        0, 2: Random trajectories from the last data-points of the current trajectories
        1: Random trajectories (At random angles) from the center

        If new_trajectory_type == 0: we generate random trajectories from the end locations of the current trajectories.
        Else, if the length(trajectories_endpoint) is < number_robots, it means the circle is not divided into
        equi-angular trajectories. We first complete the circle and then later on add random lines
        (new_trajectory_type == 1 or 2).

        :return: Location data-set
        """

        if new_trajectory_type == 0:
            new_obs, store_arr_new = self.random_observations(current_trajectories, trajectories_endpoint)
            return new_obs, store_arr_new

        if len(trajectories_endpoint) < self.n_robots:
            new_obs, store_arr_new = self.remaining_lines(current_trajectories, trajectories_endpoint)
            return new_obs, store_arr_new

        elif new_trajectory_type == 1:
            new_obs, store_arr_new = self.random_centre_lines(current_trajectories, trajectories_endpoint)
            return new_obs, store_arr_new

        else:   # new_trajectory_type == 2
            new_obs, store_arr_new = self.random_observations(current_trajectories, trajectories_endpoint)
            return new_obs, store_arr_new

    def remaining_lines(self, D_old, store_arr):
        store_arr_new = store_arr.copy()
        centre_pt = self.centre_pt
        theta = (len(store_arr) + 1) * 360 / self.n_robots
        theta = theta * np.pi / 180
        pts = D_old[len(D_old) - 1, :2].reshape(1, 2)

        for j in range(self.trajectory_length):
            x = self.data[centre_pt, 0] + np.cos(theta) * j * self.dist_x
            y = self.data[centre_pt, 1] + np.sin(theta) * j * self.dist_y
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

        for j in range(self.trajectory_length):
            x = self.data[centre_pt, 0] + np.cos(theta) * j * self.dist_x
            y = self.data[centre_pt, 1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        store_arr_new.append(pts[len(pts) - 1])
        new_obs = np.vstack((D_old[:, :2], pts))

        # plt.scatter(new_obs[:, 0], new_obs[:, 1]), plt.title("Remaining_Lines")
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return np.clip(new_obs, self.lb, self.ub), store_arr_new

    def random_observations(self, D_old, store_arr):
        store_arr_new = store_arr.copy()
        current_index = np.mod(self.iterations, len(store_arr))
        theta = np.random.randint(10, 350)
        theta = theta * np.pi / 180
        trajectory_length = np.random.randint(3, 50)
        req_pts = store_arr_new[current_index]
        pts = D_old[len(D_old) - 1, :2].reshape(1, 2)

        for j in range(trajectory_length):
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

            for j in range(self.trajectory_length):
                x = self.data[centre_pt, 0] + np.cos(theta) * j * self.dist_x
                y = self.data[centre_pt, 1] + np.sin(theta) * j * self.dist_y
                pts = np.vstack((pts, np.array([x, y])))

            store_arr.append(pts[len(pts) - 1])
            if len(pts) >= r:
                break

            pts = np.clip(pts, self.lb, self.ub)

        while len(pts) < r:
            pts, store_arr = self.random_observations(pts, store_arr)

        self.iterations = 0
        # plt.scatter(D_old[:, 0], D_old[:, 1]), plt.title(f"total_obs: {len(D_old)}, r: {r}, n_robots: {self.n_robots}")
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return pts, store_arr

    def just_one_robot(self, D_old, store_arr):
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
        trajectory_length = np.random.randint(3, 10)
        req_pts = store_arr_new[current_index]
        pts = D_old[len(D_old) - 1, :2].reshape(1, 2)

        for j in range(trajectory_length):
            x = req_pts[0] + np.cos(theta) * j * self.dist_x
            y = req_pts[1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        store_arr_new[current_index] = pts[len(pts) - 1]

        new_obs = np.vstack((D_old[:, :2], pts))
        self.robot_iterations += 1
        # plt.scatter(new_obs[:, 0], new_obs[:, 1])
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return np.clip(new_obs, self.lb, self.ub), store_arr_new