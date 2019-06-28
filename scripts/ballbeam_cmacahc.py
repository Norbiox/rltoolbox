#!/usr/bin/env python
import numpy as np

from rltoolbox.algorithm.cmac import *
from rltoolbox.environment.continuous import BallBeam
from rltoolbox.approximator import CMACApproximator
from rltoolbox.misc import compare_learning_curves, plot_learning_stats


if __name__ == "__main__":

    algorithm = CMACAHC
    environment = BallBeam(max_steps=10000, state_variables_ranges=[
        [-0.6, -0.2, 0.2, 0.6],
        [-0.6, -0.2, 0.2, 0.6]
    ])
    environment.approximate_with(CMACApproximator, n_layers=4)
    n_episodes = 10
    n_repeats = 10

    book_parameters = {'alpha': 0.005/4, 'beta': 0.005/400, 'epsilon': 0.1, 'lambd': 0.5,
                       'gamma': 0.995}

    learning_curves = np.zeros((n_repeats, n_episodes))
    for i in range(n_repeats):
        alg_instance = algorithm(environment, **book_parameters)
        alg_instance.learn(n_episodes, render=False)
        learning_curves[i, :] = alg_instance.steps_per_episode
    
    plot_learning_stats(learning_curves, 'BallBeam CMACAHC learning',
                        savefig=False)
