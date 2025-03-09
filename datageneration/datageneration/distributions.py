import numpy as np


class AbstractDistribution:
    pass


class Uniform(AbstractDistribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def logpdf(self, x):
        if x < self.a or x > self.b:
            return -np.inf  # Outside of the defined range
        else:
            return np.log(self.norm_const) - (self.alpha + 1) * np.log(x)

    def sample(self, size=1):
        u = (self.b - self.a) * np.random.uniform(0, 1, size) + self.a
        return u


class Poisson(AbstractDistribution):
    def __init__(self, lam):
        self.lam = lam

    def logpdf(self, k):
        return k * np.log(self.lam) - self.lam - np.sum(np.log(np.arange(1, k + 1)))

    def sample(self, size=tuple()):
        return np.random.poisson(self.lam, size)


class TruncatedPowerLaw(AbstractDistribution):
    def __init__(self, alpha, a, b):
        self.alpha = alpha
        self.a = a
        self.b = b

        if alpha == -1:
            self.norm_const = 1/ (np.log(self.b) - np.log(self.a))
        else:
            self.norm_const =  (self.alpha + 1) / (self.b ** (self.alpha + 1) - self.a ** (self.alpha + 1))

    def logpdf(self, x):

        inbound_term = - np.log(self.norm_const) + self.alpha * np.log(x)

        outbound_index = np.logical_and(x < self.a,  x > self.b)
        inbound_index = np.logical_not(outbound_index)

        return inbound_index * inbound_term - outbound_index * 1e80


    def sample(self, size=1):
        u = np.random.uniform(0, 1, size)
        z = u * (self.b ** (self.alpha + 1) - self.a ** (self.alpha + 1)) + self.a ** (
            self.alpha + 1
        )
        return z ** (1 / (self.alpha + 1))  # Inverse CDF sampling
