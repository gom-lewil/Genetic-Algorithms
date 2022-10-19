import numpy as np


def cross_mean(parents):
    """
    Args:
        parents (array):
            (tuples): in each tuple is all data of a parents
                individual (array):
                args* : sigma, z, ...
                fitness: fitness
    Returns:
        (tuple):
            (array): individual - mean of parents
            args* : sigma, z, ... - mean of parents
            fitness: mean fitness of parents - needs to be evaluated again
    """
    to_cross = [[] for _ in parents[0]]  # build structure to append each arg to the corresponding list
    for parent in parents:
        for i, arg in enumerate(parent):
            to_cross[i].append(arg)

    offspring = []
    for arg in to_cross:
        if type(arg[0]) == int:  # case for sigmas
            offspring.append(np.mean(arg))
        else:  # case of individuals and z
            offspring.append(np.mean(arg, axis=0))
    return offspring
