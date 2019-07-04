#!/usr/bin/env python
from rltoolbox.algorithm.classic import Q
from rltoolbox.environment.continuous import BallBeam
from rltoolbox.approximator import TableApproximator
from rltoolbox.misc import plot_learning_stats

environment = BallBeam(max_steps=10000)
environment.approximate_with(TableApproximator)
algorithm_instance = Q(environment, alpha=0.01, lambd=0.5,
                       epsilon=0.1, gamma=0.995)

algorithm_instance.learn(n_episodes=20, render=False)

plot_learning_stats(algorithm_instance.steps_per_episode,
                    title='BallBeam Q(lambda)-learning')
