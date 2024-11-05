#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/25/2020 """

import numpy as np
import torch
from time import time
from scipy.stats import norm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from BayesSwarm.util import kl_divergence_norm


class Filtering:
    def __init__(self, prediction_threshold=0.5, prediction_score=True, information_score=False):
        self.is_enabled_prediction_score = prediction_score
        self.is_enabled_information_gain_score = information_score
        self.prediction_threshold = prediction_threshold
        self.Delta = -1
        self.delta = -1
        self.jitter = 1e-10

    def filter_infomration(self, gp_model, gp_model_extended, x_star, y_star):
        is_informative = False
        self.gp_model = gp_model
        self.gp_model_extended = gp_model_extended
        if self.is_enabled_prediction_score:
            is_informative = self.prediction_score(x_star, y_star)
        if not is_informative and self.is_enabled_information_gain_score:
            is_informative = self.information_gain_score(x_star, y_star)
        
        return is_informative

    def prediction_score(self, x_star, y_star):
        is_informative = False
        delta0 = self.prediction_threshold
        if self.gp_model.get_training_status():
            mu_star, sigma_star = self.gp_model.predict(x_star)
            #self.delta = np.abs(y_star-mu_star)
            y_star = self.jitter if y_star == 0 else y_star
            self.delta = np.abs((y_star-mu_star)/y_star) 
            #print(delta, 2*delta0*sigma_star**2) 
            #if np.max(self.delta - 2*delta0*sigma_star, 0) > 0:
            if np.max(self.delta - delta0, 0) > 0:
                is_informative = True
        else:
            is_informative = True
        
        return is_informative
    
    def information_gain_score(self, x_star, y_star):
        is_informative = False

        #k = 2 #dim
        if self.gp_model.get_training_status():
            mu1, sig1 = self.gp_model(x_star)
            self.gp_model_extended.update(x_star)
            mu2, sig2 = self.gp_model_extended(x_star)

            q = (mu1, sig1)
            p = (mu2, sig2)
            #det_1 = np.linalg.det(Sigma_1)
            #det_2 = np.linalg.det(Sigma_2)
            #Sigma_2_inv = np.linalg.inv(Sigma_2)
            #trace_Simgas = np.trace(Sigma_2_inv * Sigma_1)
            #delta_mu = mu_2 - mu_1
            #mu_sig_mu = delta_mu.transpose() * Sigma_2_inv * delta_mu
            #Delta = 0.5 * (np.log(det_2/det_1) - k + trace_Simgas + mu_sig_mu)
            self.Delta = kl_divergence_norm(x_star, p, q)
        
            if self.Delta > 0.1:
                is_informative = True
        
        return is_informative

    def get_scores(self):
        score = {"predicition": self.delta, "information_gain": self.Delta}
        return score

def downsample(obs):        
    y = obs[:, 2]
    sorted_indices = np.argsort(y)[::-1]
    top_20_indices = sorted_indices[:20]    # greedy sampling of 20 observations with best signal values
    new_obs = np.delete(obs, top_20_indices, 0)
    
    model = torch.load("BayesSwarm/trvl_bestmodel.pth", map_location='cpu')
    model.eval()
    image = create_binary(new_obs)
    down_sampled = model(image)
    down_sampled = down_sampled.detach().numpy()[0]
    ds = down_sampled[:, np.newaxis, :]

    # Calculate the Euclidean distance between each point in X and Y
    distances = np.linalg.norm(ds - obs[:, :2], axis=2)

    # Find the index of the nearest point in X for each point in Y
    nearest_indices = np.argmin(distances, axis=1)
    down_sampled = obs[nearest_indices]
    down_sampled = np.vstack((down_sampled, obs[top_20_indices].reshape(20, 3)))

    return down_sampled


def create_binary(observations):          # according to create binary function of the paper
    obs, N = observations, 100
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

    # The following command will map the new observations to indices of a zero matrix representing the arena using a discretized step size.
    ij_obs = np.array([[int(abs(lb[0]) / step_size_x1) + int(abs(coord[0]) / step_size_x1) - 1 if coord[
                                                                                                      0] > 0 else int(
        abs(lb[0]) / step_size_x1) - int(abs(coord[0]) / step_size_x1),
                        int(abs(lb[1]) / step_size_x2) + int(abs(coord[1]) / step_size_x2) - 1 if coord[
                                                                                                      1] > 0 else int(
                            abs(lb[1]) / step_size_x2) - int(abs(coord[1]) / step_size_x2)]
                       for coord in obs[:, :2]])

    # ij_obs = np.clip(ij_obs, 0, 99)
    for idx, k in enumerate(ij_obs):
        x_image[k[0], k[1]] = k[0]
        y_image[k[0], k[1]] = k[1]
        blank_image[k[0], k[1]] += 1

    x_image = x_image.reshape(1, 100, 100)
    y_image = y_image.reshape(1, 100, 100)
    blank_image = blank_image.reshape(1, 100, 100)
    image = np.concatenate((blank_image, x_image, y_image))

    return torch.tensor(image, dtype=torch.float32).view(1, 3, 100, 100)

