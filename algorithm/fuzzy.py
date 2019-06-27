import numpy as np
from functools import reduce

from .abstract import FuzzyAlgorithm


def phi(environment_state):
    return np.array(reduce(lambda x, y: np.array([x * i for i in y]),
                           environment_state[::-1]))


class FQ(FuzzyAlgorithm):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, gamma=0.95,
                 alpha=0.1, *args, **kwargs):
        super().__init__(environment, lambd, epsilon, gamma, alpha,
                         *args, **kwargs)
        self.q = np.array([
            np.zeros(self.environment.approximator.state_shape)
            for i in self.actions
        ])

    def Q(self, environment_state):
        s = environment_state
        phi_s = phi(s)
        if phi_s.sum() == 0.0:
            return 0.0
        nominator = np.array([np.sum(self.q[i] * phi_s) for i in self.actions])
        return nominator / phi_s.sum()

    def get_greedy_actions(self, environment_state=None):
        s = environment_state or self.environment.state
        Q = self.Q(s)
        return np.where(Q == Q.max())[0]

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e = np.zeros(self.q.shape)

        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = r + self.gamma * self.Q(s_).max() - self.Q(s)[a]
            if self.lambd > 0.0:
                e[a] += phi(s)
                self.q += self.alpha * delta * e
                e *= self.gamma * self.lambd
            else:
                self.q[a] += self.alpha * delta * phi(s)

            if self.environment.done:
                self.environment.close()
                return self.environment


class FSARSA(FQ):

    def __init__(self, environment, lambd=0.0, gamma=0.95, alpha=0.1,
                 *args, **kwargs):
        kwargs.pop('epsilon', None)
        super().__init__(environment, lambd, None, gamma, alpha,
                         *args, **kwargs)
        self.q = np.array([
            np.zeros(self.environment.approximator.state_shape)
            for i in self.actions
        ])

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e = np.zeros(self.q.shape)
        s = self.environment.state
        a = self.get_action(epsilon_greedy=False)

        while True:
            if render:
                self.environment.render()

            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            a_ = self.get_action(epsilon_greedy=False)
            delta = r + self.gamma * self.Q(s_)[a_] - self.Q(s)[a]
            if self.lambd > 0.0:
                e[a] += phi(s)
                self.q += self.alpha * delta * e
                e *= self.gamma * self.lambd
            else:
                self.q[a] += self.alpha * delta * phi(s)

            if self.environment.done:
                self.environment.close()
                return self.environment
            s = s_
            a = a_


class FR(FQ):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, alpha=0.1,
                 beta=0.01, *args, **kwargs):
        super().__init__(environment, lambd, epsilon, None, alpha,
                         *args, **kwargs)
        self.beta = beta

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e = np.zeros(self.q.shape)
        rho = 0.0

        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = r - rho + self.Q(s_).max() - self.Q(s)[a]
            if self.lambd > 0.0:
                e[a] += phi(s)
                self.q += self.alpha * delta * e
                e *= self.lambd
            else:
                self.q[a] += self.alpha * delta * phi(s)
            if a in self.get_greedy_actions(s):
                rho += self.beta \
                    * (r - rho + self.Q(s_).max() - self.Q(s).max())

            if self.environment.done:
                self.environment.close()
                return self.environment
