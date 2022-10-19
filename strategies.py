import numpy as np
import crossover as cs
import random


def one_plus_lambda(fit_func, adam, cross=cs.cross_mean, lamb=100, ro=2, sigma=0.1, n=10, gen=100, print_results=False):
    """
    Solves the optimization problem of fit_func with the (μ + λ)-ES algorithm.

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adam (array):
        cross (func): function used to create offsprings from parents
        lamb (int): number of generated individuals per generation
        ro (int): number of parents that are used in cs.cross_mean
        sigma (float): scalar of mutation - step length
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            ((array): The best individual,
             (float): sigma of the best individual
             (float): fitness of the best individual)"""

    parent = [adam, sigma, fit_func(adam)]
    for g in range(gen):
        population = []
        for _ in range(lamb):
            # crossover & mutation
            offspring = [np.nan, sigma, 0]
            offspring[0] = parent[0] + sigma * np.random.randn(n)
            offspring[-1] = fit_func(offspring[0])
            population.append(offspring)
        # evaluation & selection
        sorted_pop = sorted(np.concatenate([population, [parent]], axis=0), reverse=False, key=lambda x: x[-1])
        parent = sorted_pop[0]

        if print_results:  # Print fitness of current generation for user
            print(f"The top result of Gen {g} is", parent[-1])
    return parent


def mu_plus_lambda(fit_func, adams, cross=cs.cross_mean, mu=20, lamb=100, ro=2, sigma=0.1, n=10, gen=100,
                   print_results=False):
    """
    Solves the optimization problem of fit_func with the (μ + λ)-ES algorithm.

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adams (array):
        cross (func): function used to create offsprings from parents
        mu (int): number of selected individuals per generation
        lamb (int): number of generated individuals per generation
        ro (int): number of parents that are used in cs.cross_mean
        sigma (float): scalar of mutation - step length
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            ((array): The best individual,
             (float): sigma of the best individual
             (float): fitness of the best individual)"""

    parents = [[i, sigma, fit_func(i)] for i in adams]
    for g in range(gen):
        population = []
        for _ in range(lamb):
            # crossover
            genes = random.sample(parents, ro)
            offspring = cross(genes)
            # mutation
            offspring[0] = offspring[0] + sigma * np.random.randn(n)
            offspring[-1] = fit_func(offspring[0])
            population.append(offspring)
        # evaluation
        ranked_pop = sorted(population + parents, reverse=False, key=lambda x: x[-1])
        # selection
        parents = ranked_pop[:mu]

        if print_results:  # Print fitness of current generation for user
            print(f"The top result of Gen {g} is", parents[0][-1])
    return parents[0]


def mu_comma_lambda(fit_func, adams, cross=cs.cross_mean, mu=20, lamb=100, ro=2, sigma=0.1, n=10, gen=100,
                    print_results=False):
    """
    Solves the optimization problem of fit_func with the (μ, λ)-ES algorithm.

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adams (array):
        cross (func): function used to create offsprings from parents
        mu (int): number of selected individuals per generation
        lamb (int): number of generated individuals per generation
        ro (int): number of parents that are used in cs.cross_mean
        sigma (float): scalar of mutation - step length
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            ((array): The best individual,
             (float): sigma of the best individual
             (float): fitness of the best individual)"""

    parents = [[i, sigma, fit_func(i)] for i in adams]
    best = parents[0]
    for g in range(gen):
        population = []
        for _ in range(lamb):
            # crossover
            genes = random.sample(parents, ro)
            offspring = cross(genes)
            # mutation
            offspring[0] = offspring[0] + offspring[1] * np.random.randn(n)
            offspring[-1] = fit_func(offspring[0])
            population.append(offspring)
        # evaluation
        ranked_pop = sorted(population, reverse=False, key=lambda x: x[-1])
        # selection
        parents = ranked_pop[:mu]
        if parents[0][-1] < best[-1]:
            best = parents[0]
        if print_results:  # Print fitness of current generation for user
            print(f"The top result of Gen {g} is", parents[0][-1])
    return best


def one_plus_one_rechenberg(fit_func, adam, sigma=0.1, d=3.3, n=10, gen=100, print_results=False):
    """Solves the optimization problem of fit_func with the (1 + 1) Rechenberg evolution strategy

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adam (array): individual
        sigma (float): of the individual
        d (float): scalar for logarithmic decrease/increase in mutation step - approximately = sqrt(n + 1)
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            ((array): The best individual,
             (float): sigma of the best individual
             (float): fitness of the best individual)"""

    parent = [adam, sigma, fit_func(adam)]
    for g in range(gen):
        # crossover & mutation
        offspring = parent[0] + parent[1] * np.random.randn(n)
        offspring = [offspring, parent[1], fit_func(offspring)]
        # evaluation & selection
        if offspring[-1] <= parent[-1]:
            parent = offspring
            parent[1] *= np.exp(4 / 5 * d)  # increase mutation step length
        else:
            parent[1] *= np.exp(-1 / 5 * d)  # decrease mutation step length
        if print_results:
            print(f"The top result of Gen {g} is", parent[-1])
    return parent


def mu_plus_lambda_rechenberg(fit_func, adams, cross=cs.cross_mean, mu=20, d=3.3, lamb=100, ro=2, sigma=0.1, n=10,
                              gen=100, print_results=False):
    """
    Solves the optimization problem of fit_func with the (μ + λ)-Rechenberg algorithm.

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adams (array): array of individuum arrays
        cross (func): function used to create offsprings from parents
        mu (int): number of selected individuals per generation
        d (float): scalar for logarithmic decrease/increase in mutation step - approximately = sqrt(n + 1)
        lamb (int): number of generated individuals per generation
        ro (int): number of parents that are used in cs.cross_mean
        sigma (float): scalar of mutation - step length
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            ((array): The best individual,
             (float): sigma of the best individual
             (float): fitness of the best individual)"""

    parents = [[i, sigma, fit_func(i)] for i in adams]

    for g in range(gen):
        population = []
        for _ in range(lamb):
            # crossover
            genes = random.sample(parents, ro)
            offspring = cross(genes)
            # mutation
            offspring[0] += sigma * np.random.randn(n)
            population.append(offspring)
        # evaluation
        ranked_pop = sorted(population + parents, reverse=False, key=lambda x: x[1])

        # Success evaluation of this generation - success, if best 5 offsprings perform better than best parent
        sorted_pop = sorted(population, reverse=False, key=lambda x: x[1])
        if np.mean([x[1] for x in sorted_pop][:4]) < parents[0][1]:
            sigma *= np.exp(4 / 5 * d)
            # print('success')
        else:
            sigma *= np.exp(-1 / 5 * d)
            # print('failure')
        # selection
        parents = ranked_pop[:mu]

        if print_results:
            print(f"The top result of Gen {g} is", parents[0][1])
    return parents[0]


def one_comma_lambda_sa(fit_func, adam, sigma=0.1, tau=0.3, lamb=100, n=10, gen=100, print_results=False):
    """
    Solves the optimization problem of fit_func with the (1, λ)Self Adaptation
    evolution strategy

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adam (array): starting individuum
        sigma (float): scalar for mutation step size
        tau (float): scalar for mutation rate - approximately = 1/sqrt(N)
        lamb (int): number of generated individuals per generation
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            (array): the overall best individual,
            (float): sigma of the best individual,
            (float): fitness of the best individual
    """

    parent = (adam, sigma, fit_func(adam))
    best = parent
    for g in range(gen):
        pop = []
        for _ in range(lamb):
            # mutation
            xi = tau * np.random.randn(1)  # positive random number scalded by tau
            z = np.random.randn(n)  # vector containing random integers which will be scaled by sigma
            sig_i = parent[1] * np.exp(xi)  # new individual sigma that is derived by old sigma scaled by e^xi
            offspring = parent[0] + sig_i * z
            pop.append((offspring, sig_i, fit_func(offspring)))  # safe the offspring and its fitness
        # evaluation & selection
        parent = sorted(pop, reverse=False, key=lambda x: x[2])[0]
        if parent[2] < best[2]:
            best = parent
        # Print results of current gen for user
        if print_results:
            print(f"The top result of Gen {g} is", parent[2])
    return best


def one_comma_lambda_derand(fit_func, adam, sigma=0.1, tau=0.3, lamb=100, d=1.3, di=10, n=10, gen=100,
                            print_results=False):
    """
    Solves the optimization problem of fit_func with the (1, λ) de-randomized evolution strategy

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adam (array): starting individuum
        sigma (float): scalar for mutation step size
        tau (float): scalar for mutation rate - approximately = 1/sqrt(N)
        lamb (int): number of generated individuals per generation
        d (float): scalar for logarithmic decrease/increase in mutation step
        di (float):
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            (array): the overall best individual,
            (float): sigma of the best individual,
            (float): fitness of the best individual
    """
    parent = (adam, sigma, fit_func(adam))
    best = parent
    for g in range(gen):
        pop = []
        for _ in range(lamb):
            # mutation
            xi = tau * np.random.randn(1)  # positive random number scalded by tau
            z = np.random.randn(n)  # vector containing random integers which will be scaled by sigma
            sig_i = parent[1] * np.exp(xi)  # new individual sigma that is derived by old sigma scaled by e^xi
            offspring = parent[0] + sig_i * z
            sig_i *= np.exp((np.linalg.norm(z) ** 2 / n - 1) / di) * np.exp(xi / d)  # de-randomization
            pop.append((offspring, sig_i, fit_func(offspring)))  # safe the offspring, its sigma, z and fitness
        # evaluation & selection
        parent = sorted(pop, reverse=False, key=lambda x: x[2])[0]
        if parent[2] < best[2]:
            best = parent
        # Print results of current gen for user
        if print_results:
            print(f"The top result of Gen {g} is", parent[2])
    return best


def one_comma_lambda_evo_path(fit_func, adam, z, lamb=100, sigma=0.1, d=1.31, c_sig=0.3, n=10, gen=100,
                              print_results=False):
    """
    Solves the optimization problem of fit_func with the (1, λ) evolution path strategy

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adam (array): starting individual,
        z (array): scaling vector for adam
        lamb (int): number of generated individuals per generation
        sigma (float): scalar of mutation - step length
        d (float): scalar for logarithmic decrease/increase in mutation step size
        c_sig (float): weight of new mutation in evolution path calculation
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            (array): the overall best individual,
            (float): z of the best individual,
            (float): fitness
    """
    epath = np.zeros(n)
    parent = [adam, z, fit_func(adam)]
    best = parent
    for g in range(gen):
        pop = []
        for i in range(lamb):
            z = np.random.randn(n)
            # crossover & mutation
            offspring = parent[0] + sigma * z
            pop.append([offspring, z, fit_func(offspring)])
        # evaluation
        parent = sorted(pop, reverse=False, key=lambda x: x[2])[0]
        epath = (1 - c_sig) * epath + c_sig * parent[1]
        sigma *= np.exp(c_sig / (2 * d) * (np.linalg.norm(epath) ** 2 / n - 1))  # line 10 of alg 7 with formula (22)

        if parent[2] < best[2]:
            best = parent
        # Print results of current gen for user
        if print_results:
            print(f"The top result of Gen {g + 1} is", parent[2])
    return best


def mu_plus_lambda_CMA(fit_func, adams, zs, cross=cs.cross_mean, mu=20, lamb=100, ro=2, sigma=0.1, n=10, gen=100,
                       c_sig=0.3, c_mu=0.1, d=1.3, print_results=False):
    """
    Solves the optimization problem of fit_func with the (μ + λ)-CMA-ES algorithm.

    Args:
        fit_func (func): evaluation function for fitness of an individual - gets optimized
        adams (array of arrays): starting population
        zs (array of arrays): zs used for adams
        cross (func): function used to create offsprings from parents
        mu (int): number of selected individuals per generation
        lamb (int): number of generated individuals per generation
        ro (int): number of parents that are used in cs.cross_mean
        sigma (float): scalar of mutation - step length
        n (int): dimensions each individual has
        gen (int): number of generations that are computed
        c_sig (float): weight of new mutation in evolution path calculation
        c_mu (float): weight of new mutation in calculation of a new C matrix
        d (float): scalar for logarithmic decrease/increase in mutation step
        print_results (bool): print the fitness of the best individual in each gen

    Return:
        (tuple):
            ((array): The best individual,
             (float): sigma of the best individual
             (float): fitness of the best individual)"""

    parents = [[k, z, fit_func(k)] for k, z in zip(adams, zs)]
    c_matrix = np.identity(n)  # correlation matrix which specifies correlations between dimensions
    evo_path = np.zeros(n)  # evolution path which scales step size adjustment
    sigmas = np.array([sigma for _ in range(n)])  # mutation step sizes for each dimension
    for g in range(gen):
        population = []
        c_sqrt = np.linalg.cholesky(c_matrix)
        for k in range(lamb):
            # crossover
            genes = random.sample(parents, ro)
            offspring = cross(genes)
            # mutation
            z = np.random.randn(n)
            offspring[0] = offspring[0] + np.array(sigmas) * c_sqrt.dot(z)
            offspring[1] = z
            offspring[-1] = fit_func(offspring[0])
            population.append(offspring)
        # evaluation
        ranked_pop = sorted(np.concatenate((population, parents), axis=0), reverse=False, key=lambda x: x[-1])
        # selection
        parents = ranked_pop[:mu]
        sum_zs = sum([x[1] for x in parents])
        for i, parent in enumerate(parents):
            parents[i][0] = parent[0] + sigmas * c_sqrt.dot(1 / mu * sum_zs)
        # Update of CMA parameters
        evo_path = (1 - c_sig) * evo_path + c_sig * 1 / mu * sum_zs
        sum_c_sqrt_z = sum([c_sqrt.dot(x[1]).dot(np.transpose(c_sqrt.dot(x[1]))) for x in parents])
        c_matrix = (1 - c_mu) * c_matrix + c_mu * (1 / mu) * sum_c_sqrt_z
        sigmas *= np.exp((c_sig / 2 * d) * (np.linalg.norm(evo_path) ** 2 / n - 1))

        if print_results:  # Print fitness of current generation for user
            print(f"The top result of Gen {g} is", parents[0][-1])
    return parents[0]
