import optimizer as o
import strategies as es
import meta_evolution as meta
import numpy as np

# gen = 1000
# mu = 20
# for n in [10, 100, 200]:
#     print(f"\n\n------------------ Now {n} Dimensions are used ---------------------------------")
#     for func in [o.sphere, o.doublesum, o.rosenbrock, o.rastrigin]:
#
#         sigma = 0.1
#         z = np.random.randn(n)
#         start_pop = np.random.randn(n) + sigma * z
#         start_pops = np.random.randn(mu, n)
#         print(f"\nNow you see {func.__name__}")
#         print(f"Result of (μ/μ, λ): "
#               f"{es.mu_comma_lambda(func, start_pops, mu=mu, ro=mu, n=n, gen=gen)[-1]}")
#         print(f"Result of (1 + λ): {es.one_plus_lambda(func, start_pop, n=n, gen=gen)[-1]}")
#         print(f"Result of (1, λ) de-randomized self adapt: "
#               f"{es.one_comma_lambda_derand(func, start_pop, d=np.sqrt(n), di=n, n=n, gen=gen)[-1]}")
#         print(f"Result of (1, λ) evolution path self adapt: "
#               f"{es.one_comma_lambda_evo_path(func, start_pop, z, c_sig=np.sqrt(1/(n+1)), d=1+np.sqrt(1/n), n=n, gen=gen)[-1]}")

# adam = np.random.randn(10)
# meta.meta_evolution(o.sphere, es.one_plus_lambda, adam)

adams = np.random.randn(20, 10)
zs = np.random.randn(20, 10)
adams += 0.1 * zs
print(es.mu_plus_lambda_CMA(o.rosenbrock, adams, zs, print_results=True, gen=1000))