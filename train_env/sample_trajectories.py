import os
import random

import numpy as np
import matplotlib.pyplot as plt


class Sampler:
    def __init__(self, resolution, obs_frequency=1, velocity=1):
        self.observation_frequency = obs_frequency
        self.velocity = velocity
        self.resolution = resolution
        self.random_travel_counter = 0
        self.robots_arr = [20, 30, 50]
        self.distance_arr = [10, 15, 22, 30, 50]

    def generate_initial_trajectory(self):
        """
        Generates Initial trajectory as seen in BayesSwarm

        """
        location_data = np.array([0, 0])#self.data[self.centre_pt, :2]
        trajectory_endpoints = []
        min_datapoints = 100

        for k in range(self.n_robots):
            theta = 1 * (k + 1) * 360 / self.n_robots
            theta = theta * np.pi / 180

            for j in range(self.trajectory_length):
                x =  np.cos(theta) * j * self.dist_x + self.centre_pt[0]
                y =  np.sin(theta) * j * self.dist_y + self.centre_pt[1]
                location_data = np.vstack((location_data, np.array([x, y])))

            trajectory_endpoints.append(location_data[len(location_data) - 1])
            if len(location_data) >= min_datapoints:
                break

            location_data = np.clip(location_data, self.lb, self.ub)

        while len(location_data) < min_datapoints:
            location_data, trajectory_endpoints = self.random_robot_travel(
                location_data, trajectory_endpoints
            )

        return location_data, trajectory_endpoints

    def reset(self, source, reset_counter):
        
        self.data, self.lb, self.ub = source.get_info()      
        self.centre_pt = [0, 0]
        self.dist_x = self.dist_y = self.observation_frequency * self.velocity

        n_robots_counter = np.int32(reset_counter / len(self.robots_arr))
        n_robots_counter = np.mod(n_robots_counter, len(self.robots_arr))
        dist_tr_counter = np.int32(reset_counter / 1)
        dist_tr_counter = np.mod(dist_tr_counter, len(self.distance_arr))

        self.n_robots = self.robots_arr[n_robots_counter]
        self.trajectory_length = self.distance_arr[dist_tr_counter]

        print(f"Robots: {self.n_robots}, dist: {self.trajectory_length}")

        location_data, trajectory_endpoints = self.generate_initial_trajectory()

        self.random_travel_counter = 0 # reset counter for the function calls of random_robot_travel()
        self.move_robot_counter = 0 # reset counter for the function calls of move_a_robot()
        self.move_current_robot = 0 # reset the robot to 0 for move_a_robot()
        
        return location_data, trajectory_endpoints

    def remaining_lines(self, previous_points, prev_traj_endpts):
        """
        completes the axes in the wheel
        """

        traj_endpts = prev_traj_endpts.copy()
        theta = (len(prev_traj_endpts) + 1) * 360 / self.n_robots
        theta = theta * np.pi / 180
        pts = previous_points[len(previous_points) - 1, :2].reshape(1, 2)

        for j in range(self.trajectory_length):
            x = self.centre_pt[0] + np.cos(theta) * j * self.dist_x
            y = self.centre_pt[1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        traj_endpts.append(pts[len(pts) - 1])
        new_obs = np.vstack((previous_points[:, :2], pts))

        return np.clip(new_obs, self.lb, self.ub), traj_endpts

    def random_first_waypoint(self, previous_points, prev_traj_endpts):
        """
        Generates trajectories at random angles from the center. the trajectories are of same length as defined by self.trajectory_length 
        during reset
        """
        traj_endpts = prev_traj_endpts.copy()
        theta = np.random.randint(1, 359)
        theta = theta * np.pi / 180
        pts = previous_points[len(previous_points) - 1, :2].reshape(1, 2)

        for j in range(self.trajectory_length):
            x = self.centre_pt[0] + np.cos(theta) * j * self.dist_x
            y = self.centre_pt[1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        traj_endpts.append(pts[len(pts) - 1])
        new_obs = np.vstack((previous_points[:, :2], pts))

        return np.clip(new_obs, self.lb, self.ub), traj_endpts

    def random_robot_travel(self, previous_points, prev_traj_endpts):
        """
        Starting with the first robot each time this function is called, successive robot moves in a random direction 
        upto certain random distance
        """
        traj_endpts = prev_traj_endpts.copy()
        current_robot = np.mod(self.random_travel_counter, len(prev_traj_endpts))
        theta = np.random.randint(10, 350)
        theta = theta * np.pi / 180
        trajectory_length = np.random.randint(3, 50)
        robot_traj_endpoint = traj_endpts[current_robot]
        pts = previous_points[len(previous_points) - 1, :2].reshape(1, 2)

        for j in range(trajectory_length):
            x = robot_traj_endpoint[0] + np.cos(theta) * j * self.dist_x
            y = robot_traj_endpoint[1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        traj_endpts[current_robot] = pts[len(pts) - 1]

        new_obs = np.vstack((previous_points[:, :2], pts))
        self.random_travel_counter += 1

        return np.clip(new_obs, self.lb, self.ub), traj_endpts

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
                x = self.centre_pt[0] + np.cos(theta) * j * self.dist_x
                y = self.centre_pt[1] + np.sin(theta) * j * self.dist_y
                pts = np.vstack((pts, np.array([x, y])))

            store_arr.append(pts[len(pts) - 1])
            if len(pts) >= r:
                break

            pts = np.clip(pts, self.lb, self.ub)

        while len(pts) < r:
            pts, store_arr = self.random_robot_travel(pts, store_arr)

        self.random_travel_counter = 0
        # plt.scatter(previous_points[:, 0], previous_points[:, 1]), plt.title(f"total_obs: {len(previous_points)}, r: {r}, n_robots: {self.n_robots}")
        # plt.xlim(-24, 24), plt.ylim(-24, 24)
        # plt.show()

        return pts, store_arr

    def move_a_robot(self, previous_points, prev_traj_endpts):
        """
        keeps simulating movement of a robot upto certain iterations (move_robot_counter)
        """
        traj_endpts = prev_traj_endpts.copy()
        if self.move_current_robot < len(prev_traj_endpts):
            if self.move_robot_counter % 15 == 0:
                self.move_current_robot += np.random.randint(1, 3)
        else:
            self.move_current_robot = 1
        current_robot = np.mod(self.move_current_robot, len(prev_traj_endpts))

        theta = np.random.randint(10, 350)
        theta = theta * np.pi / 180
        trajectory_length = np.random.randint(3, 10)
        robot_traj_endpoint = traj_endpts[current_robot]
        pts = previous_points[len(previous_points) - 1, :2].reshape(1, 2)

        for j in range(trajectory_length):
            x = robot_traj_endpoint[0] + np.cos(theta) * j * self.dist_x
            y = robot_traj_endpoint[1] + np.sin(theta) * j * self.dist_y
            pts = np.vstack((pts, np.array([x, y])))

        traj_endpts[current_robot] = pts[len(pts) - 1]

        new_obs = np.vstack((previous_points[:, :2], pts))
        self.move_robot_counter += 1

        return np.clip(new_obs, self.lb, self.ub), traj_endpts
    
    def plot_data(self, location_data):
        # limits = np.max(np.abs(location_data)) + 2
        # plt.scatter(location_data[:, 0], location_data[:, 1])
        # plt.xlim(-limits, limits), plt.ylim(-limits, limits), plt.title('sampler')
        # # plt.savefig(f'datapoints_iter_{self.iter}')
        # plt.show()
        pass