#!/usr/bin/env python3

import numpy as np

KEYS = ["x", "r", "o", "s"]
CLAZZES = ["Crosswalk", "Road Closure", "Obstacle", "Speed Limit"]
N_CLAZZES = len(CLAZZES)

# The state contains the estimated class probabilities for the cluster.
state = np.ones((N_CLAZZES, 1)) * (1.0 / N_CLAZZES) # Uniform distribution.

# The measurement matrix contains for each class (row) the probs that an specific observation is made (col).
# In our case it is quadratic because the number of classes is the number of observations.
measurement_matrix = np.array([
        [0.8, 0.2,   0,   0],
        [0.2, 0.8,   0,   0],
        [  0,   0, 0.8, 0.2],
        [  0,   0, 0.1, 0.9]])

# The system matrix defines transition probabilites between hidden states.
# In our case, we have no transitions, so it is the identity matrix.
system_matrix_transposed = np.transpose(np.identity(N_CLAZZES))

while True:
    for c in range(N_CLAZZES):
        print("({}) {:20}: {:1.2f} |{}".format(KEYS[c], CLAZZES[c], state[c, 0], "=" * int(state[c, 0] * 10 + 0.5)))
    inp = input("Observation?")
    if inp in KEYS:
        observation = KEYS.index(inp)
        # Wonham filter
        predicted = system_matrix_transposed.dot(state)
        filtered = np.multiply(measurement_matrix[:, [observation]], predicted)
        # Normalize.
        state = np.divide(filtered, sum(filtered))
