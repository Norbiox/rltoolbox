#!/usr/bin/env python
import numpy as np

from rltoolbox.algorithm.fuzzy import *
from rltoolbox.environment.continuous import BallBeam
from rltoolbox.approximator import FuzzyApproximator
from rltoolbox.misc import compare_learning_curves, plot_learning_stats


if __name__ == "__main__":

    algorithm = FQ
    environment = BallBeam(max_steps=10000)
    environment.approximate_with(FuzzyApproximator)
    n_episodes = 5
    n_repeats = 5

    book_parameters = {'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.995, 'lambd': 0.5}

    learning_curves = np.zeros((n_repeats, n_episodes))
    for i in range(n_repeats):
        alg_instance = algorithm(environment, **book_parameters)
        alg_instance.learn(n_episodes, render=False)
        learning_curves[i, :] = alg_instance.steps_per_episode
    
    plot_learning_stats(learning_curves, 'BallBeam FQ learning', savefig=False)
