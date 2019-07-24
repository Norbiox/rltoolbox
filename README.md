# RLToolBox
Reinforcement Learning Toolbox

This is a project of framework allowing to easily experiment and play with basic reinforcement learning algorithms.
RLToolBox consist of 3 main groups of components:
  * reinforcement learning algorithms
  * basic grid and continuous state environments
  * approximators, that can be used to perform experiments in continuous state environments.
  

## Installation
Unzip this package. Add newly created folder with project to PYTHONPATH environment variable.
Install all requirements with command:
```bash
pip install -r requirements.txt
```
Then you'll be able to import its modules into your scripts or Jupyter Notebooks.


## Automated tests
To run automated unit tests use command `pytest`.


## Usage
A simple example of using this framework is presented below.
```python
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

```
This code runs process of learning agent balancing ball on beam using Q-learning algorithm and table approximation of environment state values.
After all it plots results of learning (number of steps per each episode) using MatPlotLib.

Folder `scripts` contains more ready to use examples of experiments scenarios.
