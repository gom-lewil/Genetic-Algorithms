# canonical NES (with constant step size sigma)


import numpy as np
import math

N = 50

shift = np.ones(N)

def sphere(x):
	return np.dot(x,x)

fitfunction = sphere


def termination(t,fit):
	return True if (t>10000 or fit < 10e-50) else False


def cnes():

	happy = False
	sigma = 0.01
	mu_ = np.ones(N)
	t = 0
	fit = fitfunction(mu_)

	lambda_ = 100

	while not happy:

		t+=1
		happy = termination(t,fit)

		z = [np.random.randn(N) for i in range(lambda_)]
		x_= [mu_ + sigma * z[i] for i in range(lambda_)]
		fit_ = [fitfunction(x_[i]) for i in range(lambda_) ]
		pop = list(zip(z,x_,fit_))
		pop = sorted(pop, key=lambda x: x[2])
		pop=np.array(pop)

		u = [i + 5*lambda_*i**10 for i in range(lambda_)]
		u.reverse()
		all_u = sum(u)
		u = [i/all_u for i in u]
	
		# estimation of derivatives
		delta_mu_J =  np.sum([pop[i][0]*u[i] for i in range(lambda_)], axis=0)

		# gradient ascent step
		eta_mu = 0.1
		mu_ = mu_ + (eta_mu*sigma)*delta_mu_J
		fit = np.mean(pop[:lambda_,[2]],axis=0)[0]

		print(t,":",fit)

	return t


results = [cnes()]
print(results)