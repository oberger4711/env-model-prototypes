#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt

DELTA_T = 0.1
MEASUREMENT_VARIANCE = 0.05
MEASUREMENT_STD_DEV = math.sqrt(MEASUREMENT_VARIANCE)
N_MEASUREMENTS = 80
LOST_MEASUREMENTS = [30, 50] # Predict only in this interval in the form [t_start, t_end]
#LOST_MEASUREMENTS = [] # No lost measurements

def spacedmarks(x, y, Nmarks, data_ratio=None):
    import scipy.integrate
    if data_ratio is None:
        data_ratio = 1.0
    dydx = np.gradient(y, x[1])
    dxdx = np.gradient(x, x[1])*data_ratio
    arclength = scipy.integrate.cumtrapz(np.sqrt(dydx**2 + dxdx**2), x, initial=0)
    marks = np.linspace(0, max(arclength), Nmarks)
    markx = np.interp(marks, arclength, x)
    marky = np.interp(markx, x, y)
    return np.array([markx, marky])

# Lane
t = np.linspace(0, 8, 50)
points_lane_t = spacedmarks(t, np.sin(t), N_MEASUREMENTS)
points_lane = np.array(points_lane_t).T
steps = np.zeros((points_lane.shape[0], 2))
steps[1:, ] = points_lane[1:, :] - points_lane[:-1, :]
steps[0,] = steps[1,]
step_lengths = np.linalg.norm(steps, axis=1)
vs_true = step_lengths / DELTA_T
zs = points_lane + np.random.normal(0, MEASUREMENT_STD_DEV, points_lane.shape)

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
    print("Kalman Gain:\n {}".format(k_k))
    # Correct state.
    residual = (z_k - np.matmul(h, x_k))
    print(residual)
    x_k = x_k + np.matmul(k_k, residual)
    # Correct covariance.
    p_k = np.matmul((np.eye(3) - np.matmul(k_k, h)), p_k)
    print("State Covariance:\n {}".format(p_k))
    return x_k, p_k

# KF initialization
x_k = np.array([0, 0, 1.5]) # p_x, p_y, v_parallel
p_k = np.eye(3) * 10
r = np.eye(2) * MEASUREMENT_VARIANCE
variance_a = 0.6

# Run KF
xs = np.zeros((zs.shape[0], 3))
ps = np.zeros((zs.shape[0], 3, 3))
for k in range(zs.shape[0]):
    print("### Timestep {}".format(k))
    x_k, p_k = predict(x_k, p_k, variance_a, points_lane)
    print("Predicted speed: {}".format(x_k[2]))
    if len(LOST_MEASUREMENTS) != 2 or LOST_MEASUREMENTS[0] > k or LOST_MEASUREMENTS[1] < k:
        x_k, p_k = filter(x_k, p_k, zs[k, :], r, points_lane)
        print("Filtered speed: {}".format(x_k[2]))
    xs[k, :] = x_k
    ps[k, :, :] = p_k
    print()

# Plot
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_title("Position")
ax1.plot(points_lane_t[0, :], points_lane_t[1, :], "+-", label="truth")
ax1.plot(zs.T[0], zs.T[1], "+", label="measurements")
ax1.errorbar(xs.T[0], xs.T[1], yerr=np.sqrt(ps[:, 1, 1]), fmt="+-", label="estimated")
if len(LOST_MEASUREMENTS) != 2 or LOST_MEASUREMENTS[0] > k or LOST_MEASUREMENTS[1] < k:
    ax1.plot(xs.T[0, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1], xs.T[1, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1], "rx-", linewidth=3, label="prediction_only")
ax1.legend()

ax2.set_title("Velocity along Lane")
ax2.plot(xs.T[0], vs_true, "+-", label="truth")
ax2.errorbar(xs.T[0], xs.T[2], yerr=ps[:, 2, 2], fmt="+-", label="estimated")
if len(LOST_MEASUREMENTS) != 2 or LOST_MEASUREMENTS[0] > k or LOST_MEASUREMENTS[1] < k:
    xs_pred = xs.T[0, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1]
    vs_pred = xs.T[2, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1]
    err_pred = np.sqrt(ps[LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1, 2, 2])
    ax2.errorbar(xs_pred, vs_pred, yerr=err_pred, fmt="rx-", linewidth=1, label="prediction_only")
ax2.set_ylim(1.5 * min(0, np.min(vs_true)), 1.5 * max(0, np.max(vs_true)))
ax2.legend()

plt.show()
