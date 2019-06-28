#!/usr/bin/env python
import numpy as np

from rltoolbox.algorithm.classic import *
from rltoolbox.environment.grid import GRID69
from rltoolbox.misc import compare_learning_curves


if __name__ == "__main__":

    algorithms = [
        (Q, {'alpha': 0.01, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 0.3, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 0.5, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 0.7, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 0.9, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
    ]
    environment = GRID69
    n_episodes = 40
    n_repeats = 10

    histories = {}

    for alg in algorithms:
        learning_curves = np.zeros((n_repeats, n_episodes))
        for i in range(n_repeats):
            alg_instance = alg[0](environment(), **alg[1])
            alg_instance.learn(n_episodes)
            learning_curves[i, :] = alg_instance.steps_per_episode
        histories[str(alg_instance.alpha)] = learning_curves
    
    compare_learning_curves(histories, 'GRID69 learning')
