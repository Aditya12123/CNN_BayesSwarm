#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/25/2020 """
import os
import time
import numpy as np
from time import time
from scipy.stats import norm
from BayesSwarm.RANSAC import find_equidistant_points, fit
from BayesSwarm.util import kl_divergence_norm
import matplotlib.pyplot as plt
import time


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
            # self.delta = np.abs(y_star-mu_star)
            y_star = self.jitter if y_star == 0 else y_star
            self.delta = np.abs((y_star - mu_star) / y_star)
            # print(delta, 2*delta0*sigma_star**2)
            # if np.max(self.delta - 2*delta0*sigma_star, 0) > 0:
            if np.max(self.delta - delta0, 0) > 0:
                is_informative = True
        else:
            is_informative = True

        return is_informative

    def information_gain_score(self, x_star, y_star):
        is_informative = False

        # k = 2 #dim
        if self.gp_model.get_training_status():
            mu1, sig1 = self.gp_model(x_star)
            self.gp_model_extended.update(x_star)
            mu2, sig2 = self.gp_model_extended(x_star)

            q = (mu1, sig1)
            p = (mu2, sig2)
            # det_1 = np.linalg.det(Sigma_1)
            # det_2 = np.linalg.det(Sigma_2)
            # Sigma_2_inv = np.linalg.inv(Sigma_2)
            # trace_Simgas = np.trace(Sigma_2_inv * Sigma_1)
            # delta_mu = mu_2 - mu_1
            # mu_sig_mu = delta_mu.transpose() * Sigma_2_inv * delta_mu
            # Delta = 0.5 * (np.log(det_2/det_1) - k + trace_Simgas + mu_sig_mu)
            self.Delta = kl_divergence_norm(x_star, p, q)

            if self.Delta > 0.1:
                is_informative = True

        return is_informative

    def get_scores(self):
        score = {"predicition": self.delta, "information_gain": self.Delta}
        return score


def getIndices(arr, get_index, unique=False):
    get_index = get_index.tolist()
    arr = arr.tolist()
    indices = []
    if unique:
        for ele in arr:
            index = np.where(np.all(np.isclose(get_index, ele), axis=1))[0]
            indices.append(index[0])

        return indices

    for ele in arr:
        for k in ele:
            index = np.where(np.all(np.isclose(get_index, k), axis=1))[0]

            indices.append(index[0])

    return indices


def downsample(X, y, activate_set_size):

    y_idx = getIndices(np.unique(X, axis=0), X, unique=True)    # obtain indices for unique elements in X
    y = y[y_idx]
    X = np.unique(X, axis=0)
    inliers = fit(X[:, 0], X[:, 1])

    req_pts = int(activate_set_size/len(inliers))

    if req_pts <= 1:
        # If there are more lines (inliers) than points per line, sample 2 points per line
        req_pts = 2

    for k in range(len(inliers)):
        idx_inliers = getIndices(inliers[k], X, unique=True)    # obtain indices for inliers
        y_inliers = y[idx_inliers]
        pts = np.array(find_equidistant_points(inliers[k], y_inliers, num_points=req_pts))
        dwn_sampled = pts if k == 0 else np.vstack((dwn_sampled, pts))

    X_dict = {tuple(point): index for index, point in enumerate(X)}

    indices = [X_dict.get(tuple(el)) for el in dwn_sampled]
    Xe = dwn_sampled
    ye = y[indices]

    return Xe, ye

