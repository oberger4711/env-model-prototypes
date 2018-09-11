#!/usr/bin/env python3

import abc
import numpy as np

class ObstacleKF(abc.ABC):
    def __init__(self, x_0, p_0, h, q, r):
        super().__init__()
        self._x_k = x_0
        self._p_k = p_0
        self._h = h
        self._q = q
        self._r = r

    def kf_predict(self, a_k, g_k):
        # Predict state.
        self._x_k = np.matmul(a_k, self._x_k)
        # Predict covariance.
        self._p_k = np.matmul(np.matmul(a_k, self._p_k), a_k.T) + np.matmul(g_k * self._q, g_k.T)
        return self._x_k, self._p_k

    def kf_correct(self, z_k):
        h_t = self._h.T
        # Compute Kalman gain.
        k_k = np.matmul(np.matmul(self._p_k, h_t), np.linalg.inv(np.matmul(np.matmul(self._h, self._p_k), h_t) + self._r))
        print("Kalman Gain:\n {}".format(k_k))
        # Correct state.
        residual = (z_k - np.matmul(self._h, self._x_k))
        print("Residual: {}".format(residual))
        self._x_k = self._x_k + np.matmul(k_k, residual)
        # Correct covariance.
        a = np.eye(3) - np.matmul(k_k, self._h)
        self._p_k = np.matmul(np.matmul(a, self._p_k), a.T) + np.matmul(np.matmul(k_k, self._r), k_k.T)
        print("State Covariance:\n {}".format(self._p_k))
        return self._x_k, self._p_k

    @abc.abstractmethod
    def filter(self, z_k_or_none):
        pass

    def get_state(self):
        return self._x_k

    def set_state(self, state):
        assert(state.shape == self._x_k.shape)
        self._x_k = np.array(state)

    def get_covariance(self):
        return self._p_k

    def set_covariance(self, covariance):
        assert(covariance.shape == self._p_k.shape)
        self._p_k = np.array(covariance)

class FollowTrackObstacleKF(ObstacleKF):

    ACC_VARIANCE = 18000
    MEASUREMENT_VARIANCE = 0.04

    def __init__(self, delta_t, points_lane):
        x_0 = np.array([0, 0, 0]) # p_x [m], p_y [m], v_parallel [cm / s^2]
        p_0 = np.array([[50, 0, 0],
                        [0, 50, 0],
                        [0, 0, 300]])
        h = np.array([[1, 0, 0],
                      [0, 1, 0]])
        q = FollowTrackObstacleKF.ACC_VARIANCE
        r = np.eye(2) * FollowTrackObstacleKF.MEASUREMENT_VARIANCE
        super().__init__(x_0, p_0, h, q, r)
        self._delta_t = delta_t
        self._points_lane = points_lane

    def get_nearest_direction(self):
        # TODO: Interpolate?
        ds = np.linalg.norm(self._points_lane - self._x_k[:2], axis=1)
        i_nn = min(self._points_lane.shape[0] - 2, np.argmin(ds))
        d = self._points_lane[i_nn + 1] - self._points_lane[i_nn]
        d_normalized = d / np.linalg.norm(d)
        return d_normalized

    def predict(self):
        # Linearize lane at nearest neighbour.
        d = self.get_nearest_direction()
        print("Direction", d.T)
        a_k = np.array([[1, 0, (d[0] * self._delta_t) / 100],
                      [0, 1, (d[1] * self._delta_t) / 100],
                      [0, 0, 1]])
        g_k = np.array([[(d[0] * self._delta_t) / 200],
                        [(d[1] * self._delta_t) / 200],
                        [self._delta_t]])
        return self.kf_predict(a_k, g_k)

    def correct(self, z_k):
        return self.kf_correct(z_k)

    def filter(self, z_k_or_none):
        self.predict()
        if z_k_or_none is not None:
            self.correct(z_k_or_none)
        return self._x_k, self._p_k

class SteadyObstacleKF(ObstacleKF):

    PROCESS_NOISE_VARIANCE = 0.01
    MEASUREMENT_VARIANCE = 0.04

    def __init__(self):
        x_0 = np.array([0, 0, 0]) # p_x [m], p_y [m], v_parallel [cm / s^2]
        p_0 = np.array([[50, 0, 0],
                        [0, 50, 0],
                        [0, 0, 0]])
        h = np.array([[1, 0, 0],
                      [0, 1, 0]])
        q = np.eye(3) * SteadyObstacleKF.PROCESS_NOISE_VARIANCE
        r = np.eye(2) * SteadyObstacleKF.MEASUREMENT_VARIANCE
        super().__init__(x_0, p_0, h, q, r)

    def predict(self):
        a_k = np.eye(3)
        g_k = np.eye(3)
        return self.kf_predict(a_k, g_k)

    def correct(self, z_k):
        return self.kf_correct(z_k)

    def filter(self, z_k_or_none):
        self.predict()
        if z_k_or_none is not None:
            self.correct(z_k_or_none)
        return self._x_k, self._p_k

"""
Implementation of the Interacting Multiple Model (IMM) algorithm.
Check the paper "The Interacting Multiple Model Algorithm for Accurate State Estimation of Maneuvering Targets" for details.
"""
class IMMObstacleKF(ObstacleKF):

    def __init__(self, models, state_switch_matrix):
        self.__models = list(models)
        self.__num_models = len(self.__models)
        self.__prob_models = np.ones(self.__num_models) / self.__num_models
        self.__state_switch_matrix = state_switch_matrix
        self.mix_states()
        # TODO: This should only implement the kf interface.
        #super().__init__(x_0, p_0, h, q, r)

    def __get_model_states(self):
        return np.array([m.get_state() for m in models])

    def __get_model_covariances(self):
        return np.array([m.get_covariance() for m in models])

    """
    Implementation of the "State Interaction" step in the paper.
    """
    def mix_states(self):
        states = np.array(self.__get_model_states())
        covariances = np.array(self.__get_model_covariances())
        next_states = np.zeros(states.shape)
        next_covariances = np.zeros(covariances.shape)
        shape_state = states[0].shape
        for j in range(self.__num_models):
            state_next = np.zeros(shape_state)
            # Take into account every possible state transition from state i to this state j.
            probs_model_posterior = np.zeros([self.__num_models])
            for i in range(self.__num_models):
                probs_model_posterior[i] = self.__state_switch_matrix[i, j] * self.__prob_models[i]
            probs_model_posterior /= np.sum(probs_model_posterior)
            # Mix state for model j.
            next_state_j = np.sum(probs_model_posterior * states, axis=1)
            models[j].set_state(next_state_j)
            # Mix covariance for model j.
            next_covariance_j = np.sum(probs_model_posterior * (covariances + np.matmul((states - next_state), (states - next_state).T)), axis=1)
            models[j].set_covariance(next_covariance_j)

    def filter(self, z_k_or_none):
        self.mix_states()
        for m in models:
            m.filter(z_k_or_none)
        # TODO: Update model probabilities.
        # TODO: Combine state.
        return models[0].get_state(), models[0].get_covariance()

