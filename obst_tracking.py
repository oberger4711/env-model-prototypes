#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt

import obstacle_kf

DELTA_T = 0.1
MEASUREMENT_VARIANCE = 0.04
MEASUREMENT_STD_DEV = math.sqrt(MEASUREMENT_VARIANCE)
ACC_VARIANCE = 18000
N_MEASUREMENTS = 120
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
t = np.linspace(0, N_MEASUREMENTS // 10, 50)
points_lane_t = spacedmarks(t, np.sin(t), N_MEASUREMENTS)
points_lane = np.array(points_lane_t).T
steps = np.zeros((points_lane.shape[0], 2))
steps[1:, ] = points_lane[1:, :] - points_lane[:-1, :]
steps[0,] = steps[1,]
step_lengths = np.linalg.norm(steps, axis=1)
vs_true = 100 * step_lengths / DELTA_T
zs = points_lane + np.random.normal(0, MEASUREMENT_STD_DEV, points_lane.shape)

kf = obstacle_kf.FollowTrackObstacleKF(DELTA_T, points_lane)
#kf = obstacle_kf.SteadyObstacleKF()
kf = obstacle_kf.IMMObstacleKF([
    obstacle_kf.FollowTrackObstacleKF(DELTA_T, points_lane),
    obstacle_kf.SteadyObstacleKF()
    ], np.array([
        [0.5, 0.5],
        [0.5, 0.5]])

# Run KF
xs = np.zeros((zs.shape[0], 3))
ps = np.zeros((zs.shape[0], 3, 3))
for k in range(zs.shape[0]):
    print("### Timestep {}".format(k))
    if len(LOST_MEASUREMENTS) == 0 or (LOST_MEASUREMENTS[0] > k or LOST_MEASUREMENTS[1] < k):
        z_k = zs[k, :]
    else:
        z_k = None
    x_k, p_k = kf.filter(z_k)
    xs[k, :] = x_k
    ps[k, :, :] = p_k
    print()

# Plot estimations
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_title("Position")
ax1.plot(points_lane_t[0, :], points_lane_t[1, :], "+-", label="truth")
ax1.plot(zs.T[0], zs.T[1], "+", label="measurements")
#ax1.plot(xs.T[0], xs.T[1], "+-", label="estimated")
ax1.errorbar(xs.T[0], xs.T[1], xerr=np.sqrt(ps[:, 0, 0]), yerr=np.sqrt(ps[:, 1, 1]), fmt="+-", label="estimated")
if len(LOST_MEASUREMENTS) != 0:
    ax1.plot(xs.T[0, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1], xs.T[1, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1], "rx-", linewidth=3, label="prediction_only")
ax1.legend()

ax2.set_title("Velocity along Lane")
ax2.plot(xs.T[0], vs_true, "+-", label="truth")
ax2.errorbar(xs.T[0], xs.T[2], yerr=np.sqrt(ps[:, 2, 2]), fmt="+-", label="estimated")
if len(LOST_MEASUREMENTS) != 0:
    xs_pred = xs.T[0, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1]
    vs_pred = xs.T[2, LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1]
    err_pred = np.sqrt(ps[LOST_MEASUREMENTS[0]:LOST_MEASUREMENTS[1]+1, 2, 2])
    ax2.errorbar(xs_pred, vs_pred, yerr=err_pred, fmt="rx-", linewidth=1, label="prediction_only")
#ax2.set_ylim(1.5 * min(0, np.min(vs_true)), 1.5 * max(0, np.max(vs_true)))
ax2.legend()

plt.show()

