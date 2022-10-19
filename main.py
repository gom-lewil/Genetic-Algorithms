import optimizer as o
import strategies as es
import meta_evolution as meta
import numpy as np

gen = 1000
mu = 20
for n in [10, 20, 30]:
    print(f"\n\n------------------ {n} Dimensional problems  ---------------------------------")
    for func in [o.sphere, o.doublesum, o.rosenbrock, o.rastrigin]:

        sigma = 0.1
        z = np.random.randn(n)
        start_pop = np.random.randn(n) + sigma * z
        start_pops = np.random.randn(mu, n)
        print(f"\nResults for the optimization function: {func.__name__}")
        print(f"(μ/μ, λ): "
              f"{es.mu_comma_lambda(func, start_pops, mu=mu, ro=mu, n=n, gen=gen)[-1]}")
        print(f"(1 + λ): {es.one_plus_lambda(func, start_pop, n=n, gen=gen)[-1]}")
        print(f"(1, λ) de-randomized self adapt: "
              f"{es.one_comma_lambda_derand(func, start_pop, d=np.sqrt(n), di=n, n=n, gen=gen)[-1]}")
        print(f"(1, λ) evolution path self adapt: "
              f"{es.one_comma_lambda_evo_path(func, start_pop, z, c_sig=np.sqrt(1/(n+1)), d=1+np.sqrt(1/n), n=n, gen=gen)[-1]}")