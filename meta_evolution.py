import numpy as np


def meta_evolution(fit_func, strategy, adam, sigma=0.1, alpha=10, beta=10, tau=0.3, n=10, gen=100):
    """

    Args:
        fit_func (function): evaluation function for fitness of an individual - gets optimized
        strategy (function): inner evolution strategy
        adam (array): initialized individuum
        sigma (float): scalar for mutation step size
        alpha (int): number of runs of outer evolution
        beta (int): number of runs of inner evolution strategy
        tau (float): scalar for mutation rate
        n (int): dimensions of individuals
        gen (int): generations run in each evolution strategy

    Returns:
        (tuple): the best individual, its arguments and fitness
            (array): individual
            (int): arguments, like sigma, z, ...
            (float): fitness of the best individual
    """
    eve = [adam, sigma, fit_func(adam)]  # initialize starting individual
    for a in range(alpha):
        var_bar = eve[1]*np.exp(tau*np.random.randn(1))  # line 3
        best_of_beta_runs = []
        for _ in range(beta):
            start = np.random.randn(n)
            best_of_beta_runs.append(strategy(fit_func, start, sigma=var_bar, n=n, gen=gen))  # line 5 to 10 / Inner ES
        fit_var = np.median([x[-1] for x in best_of_beta_runs])  # line 13

        if fit_var < eve[-1]:
            eve = sorted(best_of_beta_runs, reverse=False, key=lambda x: x[-1])[0]
        print(f"The result of run {a+1} is {best_of_beta_runs[0][-1]}")
    return eve
