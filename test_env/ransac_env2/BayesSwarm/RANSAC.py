import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import distance
import os
import time


def fit(x, y):
    x = list(x)
    y = list(y)
    thr = 0.2  # value of threshold
    trial = 0
    total_inliers = np.zeros((1, 2))
    total_in = []
    while len(x) > 15:
        verti = False
        max = 0
        inlx = list()
        inly = list()
        arr = x
        req_inl = inlx
        for k in range(20):
            inx = list()
            iny = list()
            n = random.sample(range(0, len(x)), k=2)
            px = x[n[0]]
            py = y[n[0]]
            qx = x[n[1]]
            qy = y[n[1]]

            if px == qx:
                m = 1
                c = -px
                verti = True

            if px != qx:
                m = (py - qy) / (px - qx)
                c = py - (m * px)

            for i in range(len(x)):
                num = abs((x[i] * m) - y[i] + c)  # Numerator in formula of distance btw point and a line
                den = math.sqrt(1 + (m * m))  # Denominator in formula of distance btw point and a line
                dist = num / den
                if dist < thr:
                    inx.append(x[i])
                    iny.append(y[i])

            if len(inx) > max:
                max = len(inx)
                inlx = inx
                inly = iny
                req_inl = inlx

        trial += 1
        if verti:
            arr = y
            req_inl = inly

        not_req_indices = [arr.index(j) for j in req_inl]
        req_indices = [i for i in range(len(x)) if i not in not_req_indices]
        x = [x[i] for i in req_indices]
        y = [y[i] for i in req_indices]

        total_inliers_int = np.hstack((np.array(inlx).reshape(-1, 1), np.array(inly).reshape(-1, 1)))
        total_inliers = np.vstack((total_inliers, total_inliers_int))
        total_in.append(total_inliers_int.copy())

    total_in = np.array(total_in, dtype=object)

    return total_in


def find_equidistant_points(xarr, y_signal, num_points):
    # Step 1: Find the two furthest points
    dist_matrix = np.sqrt(np.sum((xarr[:, None] - xarr) ** 2, axis=-1))
    max_dist = np.max(dist_matrix)
    max_indices = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    p1, p2 = xarr[max_indices[0]], xarr[max_indices[1]]

    if num_points <= 1:
        max_index = np.argmax(y_signal)
        closest_points = xarr[max_index]

        return np.array(closest_points)
    # Step 2: Divide the line segment into four equal parts
    distance = max_dist / (num_points - 1)

    # Step 3: Create array of four equidistant points
    equidistant_points = np.array([p1 + i * (p2 - p1) / (num_points - 1) for i in range(num_points)])

    # Step 4: Find closest points in xarr to equidistant points
    closest_points = []
    for point in equidistant_points:
        distances = np.sqrt(np.sum((xarr - point) ** 2, axis=1))
        closest_index = np.argmin(distances)
        closest_points.append(xarr[closest_index])

    max_index = np.argmax(y_signal)
    closest_points.append(xarr[max_index])

    return np.array(closest_points)
