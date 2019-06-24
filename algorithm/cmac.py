import numpy as np

from .abstract import CMACAlgorithm


class CMACAHC(CMACAlgorithm):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, gamma=0.95,
                 alpha=0.1, beta=0.01, *args, **kwargs):
        super().__init__(environment, lambd, epsilon, gamma, alpha,
                         *args, **kwargs)
        self.beta = beta
        self.n_layers = len(self.environment.states)
        self.V, self.mi = [], []
        for layer, states in enumerate(self.environment.states):
            self.V.append(np.ones(len(states)))
            self.mi.append(np.zeros((self.V[layer].size, len(self.actions))))

    def get_greedy_actions(self, environment_state=None):
        s = environment_state or self.environment.state
        mi = np.array([self.mi[l][s[l], :] for l in range(self.n_layers)]).sum(0)
        return np.where(mi == mi.max())[0]

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e = [np.zeros(mi.shape) for mi in self.mi]

        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = [
                r + self.gamma * self.V[l][s_[l]] - self.V[l][s[l]]
                for l in range(self.n_layers)
            ]
            for l in range(self.n_layers):
                self.V[l][s[l]] += self.alpha / self.n_layers * delta[l]
                if self.lambd > 0.0:
                    e[l][s[l], a] += 1
                    self.mi[l] += self.beta * delta[l] * e[l]
                    e[l][s[l], :] *= self.gamma * self.lambd
                else:
                    self.mi[l][s[l], a] += self.beta * delta[l]

            if self.environment.done:
                self.environment.close()
                return self.environment


class CMACQ(CMACAlgorithm):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, gamma=0.95,
                 alpha=0.1, *args, **kwargs):
        super().__init__(environment, lambd, epsilon, gamma, alpha,
                         *args, **kwargs)
        self.n_layers = len(self.environment.states)
        self.q = [
            np.zeros((len(states), len(self.actions)))
            for states in self.environment.states
        ]

    def get_greedy_actions(self, environment_state=None):
        s = environment_state or self.environment.state
        Q = np.array([self.q[t][s[t], :] for t in range(self.n_layers)]).sum(0)
        return np.where(Q == Q.max())[0]

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e = [np.zeros(q.shape) for q in self.q]

        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = [
                r + self.gamma * self.q[t][s_[t], :].max() - self.q[t][s[t], a]
                for t in range(self.n_layers)
            ]
            for t in range(self.n_layers):
                if self.lambd > 0.0:
                    e[t][s[t], a] += 1
                    self.q[t][s[t], :] += self.alpha / self.n_layers * \
                        delta[t] * e[t][s[t], :]
                    e[t][s[t], :] *= self.gamma * self.lambd
                else:
                    self.q[t][s[t], a] += self.alpha / self.n_layers * delta[t]

            if self.environment.done:
                self.environment.close()
                return self.environment


class CMACSARSA(CMACQ):

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e = [np.zeros(q.shape) for q in self.q]
        s = self.environment.state
        a = self.get_action()

        while True:
            if render:
                self.environment.render()

            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            a_ = self.get_action()
            delta = [
                r + self.gamma * self.q[t][s_[t], a_] - self.q[t][s[t], a]
                for t in range(self.n_layers)
            ]
            for t in range(self.n_layers):
                if self.lambd > 0.0:
                    e[t][s[t], a] += 1
                    self.q[t][s[t], :] += self.alpha / self.n_layers * \
                        delta[t] * e[t][s[t], :]
                    e[t][s[t], :] *= self.gamma * self.lambd
                else:
                    self.q[t][s[t], a] += self.alpha / self.n_layers * delta[t]

            if self.environment.done:
                self.environment.close()
                return self.environment
            s = s_
            a = a_


class CMACR(CMACQ):

    def __init__(self, environment, lambd=0.0, epsilon=0.005, alpha=0.1,
                 beta=0.01, *args, **kwargs):
        super().__init__(environment, lambd, epsilon, None, alpha,
                         *args, **kwargs)
        self.beta = beta
        self.n_layers = len(self.environment.states)
        self.q = [
            np.zeros((len(states), len(self.actions)))
            for states in self.environment.states
        ]

    def run_learning_episode(self, render=False):
        if self.lambd > 0.0:
            e = [np.zeros(q.shape) for q in self.q]
        rho = [0.0 for t in range(self.n_layers)]

        while True:
            if render:
                self.environment.render()

            s = self.environment.state
            a = self.get_action()
            self.environment.do_action(a)
            r = self.environment.reward
            s_ = self.environment.state
            delta = [
                r - rho[t] + self.q[t][s_[t], :].max() - self.q[t][s[t], a]
                for t in range(self.n_layers)
            ]
            for t in range(self.n_layers):
                if self.lambd > 0.0:
                    e[t][s[t], a] += 1
                    self.q[t][s[t], :] += self.alpha / self.n_layers * \
                        delta[t] * e[t][s[t], :]
                    e[t][s[t], :] *= self.lambd
                else:
                    self.q[t][s[t], a] += self.alpha / self.n_layers * delta[t]
            if a in self.get_greedy_actions(s):
                for t in range(self.n_layers):
                    rho[t] += self.beta * (
                        r - rho[t]
                        + self.q[t][s_[t], :].max()
                        - self.q[t][s[t], :].max()
                    )

            if self.environment.done:
                self.environment.close()
                return self.environment
