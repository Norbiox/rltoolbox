import numpy as np

from .abstract import ClassicAlgorithm


class AHC(ClassicAlgorithm):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, gamma=0.95,
                 alpha=0.1, beta=0.01, *args, **kwargs):
        super().__init__(environment, lambd, epsilon, gamma, alpha,
                         *args, **kwargs)
        self.beta = beta
        self.V = np.ones(len(self.environment.states))
        self.mi = np.zeros((self.V.size, len(self.actions)))

    def get_greedy_actions(self, environment_state=None):
        s = environment_state or self.environment.state
        return np.where(self.mi[s, :] == self.mi[s, :].max())[0]

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e_s = np.zeros(self.V.shape)
            e_sa = np.zeros(self.mi.shape)

        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = r + self.gamma * self.V[s_] - self.V[s]
            if self.lambd > 0.0:
                e_s[s] += 1
                e_sa[s, a] += 1
                self.V += self.alpha * delta * e_s
                self.mi += self.beta * delta * e_sa
                e_s *= self.gamma * self.lambd
                e_sa *= self.gamma * self.lambd
            else:
                self.V[s] += self.alpha * delta
                self.mi[s, a] += self.beta * delta

            if self.environment.done:
                self.environment.close()
                return self.environment


class Q(ClassicAlgorithm):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, gamma=0.95,
                 alpha=0.1, *args, **kwargs):
        super().__init__(environment, lambd, epsilon, gamma, alpha,
                         *args, **kwargs)
        self.Q = np.zeros((len(self.environment.states), len(self.actions)))

    def get_greedy_actions(self, environment_state=None):
        s = environment_state or self.environment.state
        return np.where(self.Q[s, :] == self.Q[s, :].max())[0]

    def run_learning_episode(self, render=False):
        e = np.zeros(self.Q.shape)
        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = r + self.gamma * self.Q[s_, :].max() - self.Q[s, a]
            if self.lambd > 0.0:
                e[s, a] += 1
                self.Q += self.alpha * delta * e
                e *= self.gamma * self.lambd
            else:
                self.Q[s, a] += self.alpha * delta

            if self.environment.done:
                self.environment.close()
                return self.environment


class SARSA(Q):

    def __init__(self, environment, lambd=0.0, gamma=0.95, alpha=0.1,
                 *args, **kwargs):
        kwargs.pop('epsilon', None)
        super().__init__(environment, lambd, None, gamma, alpha,
                         *args, **kwargs)
        self.Q = np.zeros((len(self.environment.states), len(self.actions)))

    def run_learning_episode(self, render=False):
        e = np.zeros(self.Q.shape)
        s = self.environment.state
        a = self.get_action(epsilon_greedy=False)
        while True:
            if render:
                self.environment.render()

            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            a_ = self.get_action(epsilon_greedy=False)
            delta = r + self.gamma * self.Q[s_, a_] - self.Q[s, a]
            if self.lambd > 0.0:
                e[s, a] += 1
                self.Q += self.alpha * delta * e
                e *= self.gamma * self.lambd
            else:
                self.Q[s, a] += self.alpha * delta

            if self.environment.done:
                self.environment.close()
                return self.environment
            s = s_
            a = a_


class R(Q):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, alpha=0.1,
                 beta=0.01, *args, **kwargs):
        kwargs.pop('gamma', None)
        super().__init__(environment, lambd, epsilon, None, alpha,
                         *args, **kwargs)
        self.beta = beta
        self.Q = np.zeros((len(self.environment.states), len(self.actions)))

    def run_learning_episode(self, render=False):
        e = np.zeros(self.Q.shape)
        rho = 0.0
        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = r - rho + self.Q[s_, :].max() - self.Q[s, a]
            if self.lambd > 0.0:
                e[s, a] += 1
                self.Q += self.alpha * delta * e
                e *= self.lambd
            else:
                self.Q[s, a] += self.alpha * delta
            if a in self.get_greedy_actions(s):
                rho += self.beta \
                    * (r - rho + self.Q[s_, :].max() - self.Q[s, :].max())
            if self.environment.done:
                self.environment.close()
                return self.environment
