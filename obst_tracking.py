#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt

DELTA_T = 0.1
NOISE_VARIANCE = 0.025
N_MEASUREMENTS = 60

def spacedmarks(x, y, Nmarks, data_ratio=None):
    import scipy.integrate
    if data_ratio is None:
        data_ratio = plt.gca().get_data_ratio()
    dydx = np.gradient(y, x[1])
    dxdx = np.gradient(x, x[1])*data_ratio
    arclength = scipy.integrate.cumtrapz(np.sqrt(dydx**2 + dxdx**2), x, initial=0)
    marks = np.linspace(0, max(arclength), Nmarks)
    markx = np.interp(marks, arclength, x)
    marky = np.interp(markx, x, y)
    return np.array([markx, marky])

# Lane
t = np.linspace(0, 6, 50)
points_lane_t = spacedmarks(t, np.sin(t), N_MEASUREMENTS)
points_lane = np.array(points_lane_t).T
steps = points_lane[1, :] - points_lane[2, :]
step_lengths = np.linalg.norm(steps)
v_true = np.average(step_lengths) / DELTA_T
zs = points_lane + np.random.normal(0, math.sqrt(NOISE_VARIANCE), points_lane.shape)

def get_nearest_direction(x_k, points_lane):
    ds = np.linalg.norm(points_lane - x_k[:2], axis=1)
    i_nn = min(points_lane.shape[0] - 2, np.argmin(ds))
    d = points_lane[i_nn + 1] - points_lane[i_nn]
    d_normalized = d / np.linalg.norm(d)
    return d_normalized

def predict(x_k, p_k, variance_a, points_lane):
    # Get direction of nearest neighbour of lane poly chain.
    d = get_nearest_direction(x_k, points_lane)

    # TODO: Interpolate.
    a = np.array([[1, 0, d[0] * DELTA_T],
                  [0, 1, d[1] * DELTA_T],
                  [0, 0, 1]])
    g_k = np.array([[(d[0] * DELTA_T) / 2],
                    [(d[1] * DELTA_T) / 2],
                    [DELTA_T]])
    g_k_t = g_k.T
    q_k = variance_a

    # Predict state (linearized).
    x_k1 = np.matmul(a, x_k)

    # TODO: Predict state (EKF).
    # TODO: Accurately accumulate point to point distances until the estimated distance is passed.
    #x_k1 = x_k
    #p_lane_currents = points_lane[i_nn]
    #d_left = x_k[2] * DELTA_T
    #while i_current < points_lane.shape[0] && d_left > 
    #x_k1 = x_k +

    # Predict covariance.
    p_k1 = np.matmul(a, p_k) + np.matmul(g_k * q_k, g_k_t)

    return x_k1, p_k1

def filter(x_k, p_k, z_k, r, points_lane):
    h = np.array([[1, 0, 0],
                  [0, 1, 0]])
    h_k_t = h.T
    # Compute Kalman gain.
    k_k = np.matmul(np.matmul(p_k, h_k_t), np.linalg.inv(np.matmul(np.matmul(h, p_k), h_k_t) + r))
    print("Kalman Gain: {}".format(k_k.T))
    # Correct state.
    residual = (z_k - np.matmul(h, x_k))
    print(residual)
    x_k = x_k + np.matmul(k_k, residual)
    # Correct covariance.
    p_k = np.matmul((np.eye(3) - np.matmul(k_k, h)), p_k)
    return x_k, p_k

# KF initialization
x_k = np.array([0, 0, 0.5]) # p_x, p_y, v_parallel
p_k = np.eye(3) * 1000
r = np.eye(2) * NOISE_VARIANCE
variance_a = 0.2

# Run KF
xs = np.zeros((zs.shape[0], 3))
for k in range(zs.shape[0]):
    x_k, p_k = predict(x_k, p_k, variance_a, points_lane)
    print("Predicted speed: {}".format(x_k[2]))
    x_k, p_k = filter(x_k, p_k, zs[k, :], r, points_lane)
    xs[k, :] = x_k
    print("Filtered speed: {}".format(x_k[2]))
    print()

# Plot
plt.plot(points_lane_t[0, :], points_lane_t[1, :], "+-", label="true")
plt.plot(zs.T[0], zs.T[1], "+", label="measurements")
plt.plot(xs.T[0], xs.T[1], "+-", label="position_estimated")
plt.plot(xs.T[0], xs.T[2], "+-", label="velocity_estimated")
plt.plot(xs.T[0], np.ones((xs.shape[0])) * v_true, "+-", label="velocity_true")
plt.legend()

plt.show()
