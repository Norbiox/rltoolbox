#!/usr/bin/env python
import numpy as np

from rltoolbox.algorithm.cmac import *
from rltoolbox.environment.continuous import BallBeam
from rltoolbox.approximator import CMACApproximator
from rltoolbox.misc import compare_learning_curves, plot_learning_stats


if __name__ == "__main__":

    algorithm = CMACQ
    environment = BallBeam(max_steps=10000)
    environment.approximate_with(CMACApproximator, n_layers=4)
    n_episodes = 10
    n_repeats = 5

    book_parameters = {'alpha': 1, 'epsilon': 0.1, 'gamma': 0.995, 'lambd': 0.0}

    learning_curves = np.zeros((n_repeats, n_episodes))
    for i in range(n_repeats):
        alg_instance = algorithm(environment, **book_parameters)
        alg_instance.learn(n_episodes, render=False)
        learning_curves[i, :] = alg_instance.steps_per_episode
    
    plot_learning_stats(learning_curves, 'BallBeam CMACQ learning', savefig=False)
