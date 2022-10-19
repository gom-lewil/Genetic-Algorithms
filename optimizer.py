import numpy as np


def sphere(x):
    return np.dot(x, x)


def doublesum(x):
    return sum([sum(x[:i]) ** 2 for i, _ in enumerate(x)])


def rosenbrock(x):
    return sum([100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])


def rastrigin(x, a=10):
    return a * len(x) + sum([i ** 2 + a * np.cos(2 * np.pi * i) for i in x])
