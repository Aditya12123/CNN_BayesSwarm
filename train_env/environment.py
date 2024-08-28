import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sample_trajectories import Sampler
from source import Source
from geomloss import SamplesLoss


class RobotEnv:
    def __init__(self, batch_size):
        self.trajectory_endpoints = None
        self.observations = None
        self.reset_counter = 0
        self.iter = 0
        self.num_images = 0
        self.resolution = 100
        self.termination_iter = 600
        self.batch_size = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.source = Source(self.resolution)
        self.mse_loss = nn.MSELoss()
        self.sampler = Sampler(self.resolution)

    def step(self):
        """
        Reset the sampler after certain number of iterations. Each time the sampler is reset, number_robots, 
        distance_travelled_per_trajectory changes

        :return: Images of size [self.batch_size, 100, 100]
        """

        while self.num_images < self.batch_size:
            self.iter += 1
            self.num_images += 1
            termination_lines = 4 * self.sampler.n_robots

            if self.iter > 400:
                if self.iter < 430 or self.iter > 585:
                    self.observations, self.trajectory_endpoints = self.sampler.initial_shape_reset()
                    title = 'initial_shape_reset'

                elif 430 <= self.iter < 430 + self.sampler.n_robots:
                    self.observations, self.trajectory_endpoints = self.sampler.remaining_lines(self.observations, self.trajectory_endpoints)
                    title = 'remaining_lines'

                else:
                    self.observations, self.trajectory_endpoints = self.sampler.move_a_robot(self.observations, self.trajectory_endpoints)
                    title = 'move_a_robot'

            else:

                if self.iter == 1 or self.iter == termination_lines + 150:
                    self.observations, self.trajectory_endpoints = self.sampler.reset(self.source, self.reset_counter - 1)
                    title = 'reset'
                
                elif len(self.trajectory_endpoints) < self.sampler.n_robots:
                    self.observations, self.trajectory_endpoints = self.sampler.remaining_lines(self.observations, self.trajectory_endpoints)
                    title = 'remaining_lines'
                
                elif self.iter < termination_lines:
                    self.observations, self.trajectory_endpoints = self.sampler.random_first_waypoint(self.observations, self.trajectory_endpoints)
                    title = 'random_first_waypoint'

                else:
                    self.observations, self.trajectory_endpoints = self.sampler.random_robot_travel(self.observations, self.trajectory_endpoints)
                    title = 'random_robot_travel'

            if self.num_images == 1:
                self.total_obs = []
                self.total_obs.append(self.observations)
                mtx = self.create_binary(self.observations)
            else:
                self.total_obs.append(self.observations)
                mtx = torch.concat((mtx, self.create_binary(self.observations)), dim=0)
            
            if self.iter >= self.termination_iter:
                self.reset()
            
            path = f"/data1/users/abhatt4/cnn_bayesswarm/CNN_BayesSwarm_RAL/assets/robots_{self.sampler.n_robots}_trajlen_{self.sampler.trajectory_length}" #'F:\\ADAMS_Lab\\CCR_Train\\cnn_train_env\\assets\\images'
            if not os.path.exists(path):
                os.makedirs(path)
            limits = np.max(np.abs(self.observations)) + 5
            plt.scatter(self.observations[:, 0], self.observations[:, 1])
            plt.xlim(-limits, limits), plt.ylim(-limits, limits), plt.title(title + str(self.iter))
            plt.savefig( path + f'/data_iter_{self.iter}.png')
            # plt.show()

            # print(self.observations.shape)
        self.num_images = 0


        return mtx

    def reset(self):
        self.iter = 0
        self.source.generate_arena(self.reset_counter)
        self.data, self.lb, self.ub = self.source.get_info()
        self.observations, self.trajectory_endpoints = self.sampler.reset(self.source, self.reset_counter)
        self.reset_counter += 1

    def create_binary(self, observations):
        """
        Implementation of GenerateMatrix Function of our paper

        :param observations: The location data-points collected by the robots
        :return: [X, Y, B] matrix 
        
        """
        obs, N = observations, self.resolution
        limit = np.ceil(np.max(np.abs(obs))) + 5
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

        :return: Nearest observations among the input observations to the down-sampled observations in the 
                 euclidean space
                 
        """
        distances = torch.norm(model_output.view(1, 100, 2) - input_observations[:, :2].view(input_observations.shape[0], 1, 2), dim=2)
        nearest_indices = torch.argmin(distances, dim=0)

        return input_observations[nearest_indices, :2]

    def loss(self, down_sampled):
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=2)
        dwn_sample_loss = 0

        for k in range(len(self.total_obs)):
            obs = torch.tensor(self.total_obs[k], dtype=torch.float32).to(self.device)
            dwn = down_sampled[k]
            # print(obs.shape, dwn.shape)
            dwn_totalSubset_xy = self.nearest_points(obs, model_output=dwn)
            dwn_sample_loss += sinkhorn_loss(obs[:, :2], dwn[:, :2])
            dwn_sample_loss += self.mse_loss(dwn[:, :2], dwn_totalSubset_xy)
        
        dwn_sample_loss = dwn_sample_loss / self.batch_size

        loss = dwn_sample_loss
        return loss
