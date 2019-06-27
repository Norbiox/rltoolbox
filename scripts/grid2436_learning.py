#!/usr/bin/env python
import numpy as np

from rltoolbox.algorithm.classic import *
from rltoolbox.environment.grid import GRID2436
from rltoolbox.misc import compare_learning_curves


if __name__ == "__main__":

    algorithms = [
        #(Q, {'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        #(Q, {'alpha': 0.3, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        #(Q, {'alpha': 0.5, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        #(Q, {'alpha': 0.7, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        #(Q, {'alpha': 0.9, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 1.0, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 1.1, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
        (Q, {'alpha': 1.2, 'epsilon': 0.1, 'gamma': 0.95, 'lambd': 0.0}),
    ]
    environment = GRID2436
    n_episodes = 200
    n_repeats = 5

    histories = {}

    for alg in algorithms:
        learning_curves = np.zeros((n_repeats, n_episodes))
        for i in range(n_repeats):
            alg_instance = alg[0](environment(max_steps=10000), **alg[1])
            print(f"{alg_instance.name} {alg_instance.environment.name} alpha = {alg[1]['alpha']}, repetition: {i}")
            alg_instance.learn(n_episodes, print_status=False)
            learning_curves[i, :] = alg_instance.steps_per_episode
        histories[alg[1]['alpha']] = learning_curves
    
    compare_learning_curves(histories, 'GRID2436 Q-learning - uczenie się z różnymi wartościami parametru alpha')
