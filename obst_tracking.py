#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

DELTA_T = 0.1
NOISE_VARIANCE = 0.3
N_MEASUREMENTS = 20

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
t = np.linspace(0, 2, 50)
points_lane_t = spacedmarks(t, np.sin(t), N_MEASUREMENTS)
points_lane = np.array(points_lane_t).T
zs = points_lane + np.random.normal(0, NOISE_VARIANCE ** 2, points_lane.shape)

def predict(x_k, p_k, q, points_lane):
    # Get nearest neighbour of lane poly chain.
    ds = np.linalg.norm(points_lane - x_k[:2], axis=1)
    i_nn = min(points_lane.shape[0] - 2, np.argmin(ds))

    # TODO: Interpolate.
    d = points_lane[i_nn + 1] - points_lane[i_nn]
    d_normalized = d / np.linalg.norm(d)
    a = np.array([[1, 0, d_normalized[0] * DELTA_T], [0, 1, d_normalized[1] * DELTA_T], [0, 0, 1]])

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
    p_k1 = np.matmul(a, p_k) + q

    return x_k1, p_k1

def filter(x_k, p_k, z_k, r, points_lane):
    h = np.array([[1, 0, 0], [0, 1, 0]])
    h_k_t = h.T
    # Compute Kalman gain.
    k_k = np.matmul(np.matmul(p_k, h_k_t), np.linalg.inv(np.matmul(np.matmul(h, p_k), h_k_t) + r))
    # Correct state.
    residual = (z_k - np.matmul(h, x_k))
    print(residual)
    x_k = x_k + np.matmul(k_k, residual)
    # Correct covariance.
    p_k = np.matmul((np.eye(3) - np.matmul(k_k, h)), p_k)
    return x_k, p_k

# EKF initialization
x_k = np.array([0, 0, 1.2]) # p_x, p_y, v_parallel
p_k = np.eye(3) * 10
r = np.eye(2) * 2
q = np.eye(3) * 0.1

xs = np.ones((zs.shape[0], 3))
for k in range(zs.shape[0]):
    x_k, p_k = predict(x_k, p_k, q, points_lane)
    print("Predicted speed: {}".format(x_k[2]))
    x_k, p_k = filter(x_k, p_k, zs[k, :], r, points_lane)
    xs[k, :] = x_k
    print("Filtered speed: {}".format(x_k[2]))
    print()

# Plot
plt.plot(points_lane_t[0, :], points_lane_t[1, :], label="true")
plt.plot(xs.T[0], xs.T[1], label="estimation")
plt.legend()

plt.show()
