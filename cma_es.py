import array

import numpy as np
import matplotlib.pyplot as plt


def sphere(x):
    return np.dot(x, x)


def fitness(x):
    return sphere(x)


def cma_es(dimension, mu, lamb, sigma_val, generations):
    c_sig = np.sqrt(1/(dimension + 1))
    c_mu = 1 / np.sqrt(dimension**2 + 1)
    s_sig = np.zeros(dimension)
    d = 1 + np.sqrt(1/dimension)
    C = 1
    sigma = np.zeros(dimension)
    for i in range(dimension):
        sigma[i] = sigma_val
    x = np.random.randn(dimension)
    results = []

    curr_gen = 0
    pop = []
    while curr_gen < generations:
        curr_gen += 1
        children = []
        for i in range(lamb):
            z_k = np.random.randn(dimension)
            x_k = x + sigma * np.sqrt(C) * z_k
            children.append((x_k, fitness(x_k), z_k))
        pop = sorted(children + pop, key=lambda tup: tup[1])[:mu]
        z_ks = sum([vals[2] for vals in pop]) / mu
        x = x + sigma * np.sqrt(C) * z_ks
        s_sig = (1 - c_sig) * s_sig + c_sig * z_ks
        C_z_k_sum = sum(np.sqrt(C) * vals[2] * np.transpose(np.sqrt(C) * vals[2]) for vals in pop)
        C = (1 - c_mu) * C + c_mu * (C_z_k_sum / mu)
        sigma = sigma * np.exp((((np.linalg.norm(s_sig)**2) / dimension) - 1) * (c_sig / (2*d)))

        print(f"Generation: {curr_gen}, Best fitness: {pop[0][1]}")
        results.append(pop[0][1])
    plt.plot(range(len(results)), results)
    #plt.yscale("log")
    #plt.xscale("log")
    plt.show()


cma_es(1000, 20, 20, .2, 1000)


