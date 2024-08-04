import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from new_sampler import Sampler
from robot_source import Source
from geomloss import SamplesLoss


class RobotEnv:
    def __init__(self, batch_size):
        self.trajectory_endpoints = None
        self.last_state = None
        self.D_new = None
        self.observations = None
        self.reset_counter = 0
        self.iter = 0
        self.num_images = 0
        self.resolution = 100
        self.termination_iter = 600
        self.batch_size = batch_size
        self.source = Source(self.resolution)
        self.mse_loss = nn.MSELoss()
        self.sampler = Sampler(self.resolution, obs_frequency=1, velocity=1)
        self.max_length = [0, 0]

    def step(self, epoch=0):
        """
        Reset the sampler after certain number of iterations. Each time the sampler is reset, number_robots, 
        distance_travelled_per_trajectory changes

        :param epoch:
        :return: Image of size [self.batch_size, 100, 100]
        """

        self.epoch = epoch

        while self.num_images < self.batch_size:
            self.iter += 1
            self.num_images += 1
            termination_lines = 4 * self.sampler.n_robots
            self.just_random = 1 if self.iter < termination_lines else 2

            if self.iter > 400:
                if self.iter < 430 or self.iter > 585:
                    self.observations, self.trajectory_endpoints = self.sampler.initial_shape_reset()
                elif 430 <= self.iter < 430 + self.sampler.n_robots:
                    self.observations, self.trajectory_endpoints = self.sampler.remaining_lines(self.observations, self.trajectory_endpoints)
                else:
                    self.observations, self.trajectory_endpoints = self.sampler.just_one_robot(self.observations, self.trajectory_endpoints)

            else:
                if self.iter == 1 or self.iter == termination_lines + 150:
                    self.observations, self.trajectory_endpoints = self.sampler.reset(self.source, self.reset_counter - 1)
                else:
                    self.observations, self.trajectory_endpoints = self.sampler.generate_new_trajectories(self.observations, self.trajectory_endpoints,
                                                                                        self.just_random)

            if self.num_images == 1:
                self.total_obs = self.observations
                self.number_observations = []
            else:
                self.total_obs = np.vstack((self.total_obs, self.observations))

            self.number_observations.append(self.total_obs.shape[0])
            done = True if self.iter >= self.termination_iter else False
            if done:
                self.reset()

        image = self.get_Image()
        self.num_images = 0

        return image

    def reset(self):
        self.iter = 0
        self.source.generate_arena(self.reset_counter)
        self.data, self.lb, self.ub = self.source.get_info()
        self.observations, self.trajectory_endpoints = self.sampler.reset(self.source, self.reset_counter)
        self.reset_counter += 1

    def get_Image(self):

        for k in range(len(self.number_observations)):
            if k == 0:
                obs = self.total_obs[:self.number_observations[k]]
                mtx = self.create_binary(obs)
            else:
                obs = self.total_obs[self.number_observations[k - 1] + 1:self.number_observations[k]]
                mtx = torch.concat((mtx, self.create_binary(obs)), dim=0)

        return mtx

    def create_binary(self, observations):
        """
        
        :param observations: 
        :return: [X, Y, B] matrix according to our paper
        
        """
        obs, N = observations, self.resolution
        limit = np.ceil(np.max(obs)) + 5
        lb = [-limit, -limit]
        ub = [limit, limit]
        if np.size(obs) == 0:
            return np.zeros((N, N))

        blank_image = np.zeros((N, N))
        x_image = np.full((N, N), -limit)
        y_image = np.full((N, N), -limit)
        step_size_x1 = (abs(lb[0]) + abs(ub[0])) / N
        step_size_x2 = (abs(lb[1]) + abs(ub[1])) / N

        # The following command will map the new observations to indices of a zero matrix 
        # representing the arena using a discretized step size.
        
        ij_obs = np.array([[int(abs(lb[0]) / step_size_x1) + int(abs(coord[0]) / step_size_x1) - 1 if coord[
                                                                                                          0] > 0 else int(
            abs(lb[0]) / step_size_x1) - int(abs(coord[0]) / step_size_x1),
                            int(abs(lb[1]) / step_size_x2) + int(abs(coord[1]) / step_size_x2) - 1 if coord[
                                                                                                          1] > 0 else int(
                                abs(lb[1]) / step_size_x2) - int(abs(coord[1]) / step_size_x2)]
                           for coord in obs[:, :2]])

        for idx, k in enumerate(ij_obs):
            try:
                x_image[k[0], k[1]] = k[0]
            except IndexError as e:
                print(f"IndexError: {e}, max: {limit}")
                print(ij_obs[idx])
                print(obs[idx])
                print(lb)
                print(ub)
                print(step_size_x1, step_size_x2)
                
            y_image[k[0], k[1]] = k[1]
            blank_image[k[0], k[1]] += 1

        x_image = x_image.reshape(1, N, N)
        y_image = y_image.reshape(1, N, N)
        blank_image = blank_image.reshape(1, N, N)
        image = np.concatenate((blank_image, x_image, y_image))

        return torch.tensor(image, dtype=torch.float32).view(1, 3, N, N)

    def nearest_points(self, input_observations, model_output):
        """
        Implementation of NearestPoints function of Algorithm1

        :param input_observations: Input Observations
        :param model_output: Down-sampled observations that the model outputs
        :return: Nearest observations among the input observations to the down-sampled observations in the 
                 euclidean space
                 
        """
        distances = torch.norm(model_output.view(1, 100, 2) - input_observations[:, :2].view(input_observations.shape[0], 1, 2), dim=2)
        nearest_indices = torch.argmin(distances, dim=0)

        return input_observations[nearest_indices, :2]

    def loss(self, down_sampled):
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=2)
        observations = torch.tensor(self.total_obs, dtype=torch.float32).to('cpu')
    
        for k in range(len(self.number_observations)):
            if k == 0:
                obs = observations[:self.number_observations[k]]
                dwn = down_sampled[k]
                dwn_totalSubset_xy = self.nearest_points(obs, model_output=dwn[:, :2])
                s_l = sinkhorn_loss(obs[:, :2], dwn[:, :2])
                m_l = self.mse_loss(dwn[:, :2], dwn_totalSubset_xy)
                dwn_sample_loss = s_l + m_l
            else:
                obs = observations[self.number_observations[k - 1]:self.number_observations[k]]
                dwn = down_sampled[k]
                dwn_totalSubset_xy = self.nearest_points(obs, model_output=dwn)
                dwn_sample_loss += sinkhorn_loss(obs[:, :2], dwn[:, :2])
                dwn_sample_loss += self.mse_loss(dwn[:, :2], dwn_totalSubset_xy)
    
        dwn_sample_loss = dwn_sample_loss / self.batch_size

        loss = dwn_sample_loss
        return loss
