#!/usr/bin/env python3

import abc
import numpy as np
import math
import scipy.stats

class IObstacleKF(abc.ABC):

    @abc.abstractmethod
    def filter(self, z_k_or_none):
        pass

    @abc.abstractmethod
    def get_state(self):
        pass

    @abc.abstractmethod
    def set_state(self, state):
        pass

    @abc.abstractmethod
    def get_covariance(self):
        pass

    @abc.abstractmethod
    def set_covariance(self, covariance):
        pass

    @abc.abstractmethod
    def get_likelihood(self):
        pass

class ObstacleKF(IObstacleKF):
    def __init__(self, x_0, p_0, h, q, r):
        super().__init__()
        self._x_k = x_0 # State
        self._p_k = p_0 # Covariance
        self._p_k_predicted = p_0 # Covariance
        self._h = h # Observation model
        self._q = q # Process noise
        self._r = r # Observation noise
        self._z_k = None
        self._z_k_pred = np.dot(h, x_0) # Predicted observation of previous update
        self._s_k = None

    def kf_predict(self, a_k, g_k):
        # Predict state.
        self._x_k = np.dot(a_k, self._x_k)
        # Predict covariance.
        self._p_k = np.dot(np.dot(a_k, self._p_k), a_k.T) + np.dot(g_k * self._q, g_k.T)
        self._p_k_predicted = self._p_k
        self._z_k_pred = np.dot(self._h, self._x_k)
        return self._x_k, self._p_k

    def kf_correct(self, z_k):
        self._z_k = z_k.copy()
        h_t = self._h.T
        # Compute Kalman gain.
        self._s_k = np.dot(np.dot(self._h, self._p_k), h_t) + self._r
        k_k = np.dot(np.dot(self._p_k, h_t), np.linalg.inv(self._s_k))
        # Correct state.
        self._z_k_pred = np.dot(self._h, self._x_k)
        innovation = (z_k - self._z_k_pred)
        self._x_k = self._x_k + np.dot(k_k, innovation)
        # Correct covariance.
        a = np.eye(3) - np.dot(k_k, self._h)
        self._p_k = np.dot(np.dot(a, self._p_k), a.T) + np.dot(np.dot(k_k, self._r), k_k.T)
        return self._x_k, self._p_k

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

    def get_prediction(self):
        return self._z_k_pred

    def get_innovation_covariance(self):
        return self._s_k

    def get_likelihood(self):
        return scipy.stats.multivariate_normal.pdf(self._z_k, self._z_k_pred, self._s_k)

class FollowTrackObstacleKF(ObstacleKF):

    ACC_VARIANCE = 180
    MEASUREMENT_VARIANCE = 0.08

    def __init__(self, delta_t, points_lane):
        x_0 = np.array([0, 0, 100]) # p_x [m], p_y [m], v_parallel [cm / s^2]
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
        print("Estimated speed: {}".format(self._x_k[2]))
        self.predict()
        self._x_k[2] = max(40, self._x_k[2])
        if z_k_or_none is not None:
            self.correct(z_k_or_none)
        print("Estimated speed: {}".format(self._x_k[2]))
        return self._x_k, self._p_k

class SteadyObstacleKF(ObstacleKF):

    PROCESS_NOISE_VARIANCE = 0.01
    MEASUREMENT_VARIANCE = 0.08

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
class IMMObstacleKF(IObstacleKF):

    def __init__(self, models, state_switch_matrix):
        self.__models = list(models)
        self.__num_models = len(self.__models)
        #self.__prob_models = np.ones(self.__num_models) / self.__num_models
        self.__prob_models = np.array([0.5, 0.5])
        self.__state_switch_matrix = state_switch_matrix
        self.__x_k, self.__p_k = self.combine_state()

    def __get_model_states(self):
        return np.array([m.get_state() for m in self.__models])

    def __get_model_covariances(self):
        return np.array([m.get_covariance() for m in self.__models])

    """
    Implementation of the "State Interaction" step in the paper.
    """
    def mix_states(self):
        states = np.array(self.__get_model_states())
        covariances = np.array(self.__get_model_covariances())
        next_states = np.zeros(states.shape)
        next_covariances = np.zeros(covariances.shape)
        shape_state = states[0].shape
        shape_covariance = covariances[0].shape
        for j in range(self.__num_models):
            state_next = np.zeros(shape_state)
            # Take into account every possible state transition from state i to this state j.
            probs_model_posterior = np.zeros([self.__num_models])
            for i in range(self.__num_models):
                probs_model_posterior[i] = self.__state_switch_matrix[i, j] * self.__prob_models[i]
            probs_model_posterior /= np.sum(probs_model_posterior)
            # Mix state for model j.
            next_state_j = np.zeros(shape_state)
            for i in range(self.__num_models):
                next_state_j += probs_model_posterior[i] * states[i]
            self.__models[j].set_state(next_state_j)
            # Mix covariance for model j.
            next_covariance_j = np.zeros(shape_covariance)
            for i in range(self.__num_models):
                diff = states[i] - next_state_j
                next_covariance_j += probs_model_posterior[i] * (covariances[i] + np.dot(diff, diff.T))
            # Mix covariance for model j.
            self.__models[j].set_covariance(next_covariance_j)

    """
    Implementation of the "Model Probability Update" step in the paper.
    """
    def update_probs(self, z_k):
        # Update.
        next_prob_models = self.__prob_models.copy()
        for j in range(self.__num_models):
            model_j = self.__models[j]
            likelihood_j = model_j.get_likelihood()
            prob_state_j = sum(self.__state_switch_matrix[i, j] * self.__prob_models[i] for i in range(self.__num_models))
            next_prob_models[j] = likelihood_j * prob_state_j
        # Normalize.
        next_prob_models /= sum(p for p in next_prob_models)
        print("Model Probabilities:")
        for j in range(self.__num_models):
            print("  {}: {}".format(self.__models[j].__class__.__name__, self.__prob_models[j]))
        self.__prob_models = next_prob_models

    """
    Implementation of the "State Estimate Combination" step in the paper.
    """
    def combine_state(self):
        states = self.__get_model_states()
        covariances = self.__get_model_covariances()
        next_states = np.zeros(states.shape)
        next_covariances = np.zeros(covariances.shape)
        shape_state = states[0].shape
        shape_covariance = covariances[0].shape
        # Combine states as weighted sum.
        state_combined = np.zeros(shape_state)
        for j in range(self.__num_models):
            state_combined += self.__prob_models[j] * states[j]
        # Combine covariances.
        covariance_combined = np.zeros(shape_covariance)
        for j in range(self.__num_models):
            diff = states[j] - state_combined
            covariance_combined += self.__prob_models[j] * (covariances[j] + np.dot(diff, diff.T))
        return state_combined, covariance_combined

    def filter(self, z_k_or_none):
        self.mix_states()
        for m in self.__models:
            m.filter(z_k_or_none)
        if z_k_or_none is not None:
            self.update_probs(z_k_or_none)
        self.__x_k, self.__p_k = self.combine_state()
        return self.__x_k, self.__p_k

    def get_state(self):
        return self.__x_k

    def set_state(self, state):
        self._x_k = state.copy()

    def get_covariance(self):
        return self.__p_k

    def set_covariance(self, covariance):
        self.__p_k = covariance.copy()

    def get_likelihood(self):
        return np.array([m.get_likelihood() for m in self.__models])
