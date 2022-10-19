# main ES for lecture "Evolution Strategies", Oliver Kramer, 2021/22

import numpy as np
import math

N = 1000

shift = np.ones(N)

def shifted_sphere(x):
	return np.dot(x-shift,x-shift)

def sphere(x):
	return np.dot(x,x)

fitfunction = sphere

def termination(t,fit):
	return True if (t>1000 or fit < 10e-50) else False


# (1+1)-ES with Rechenberg rule, Chapters 1+3

def oneplusone():

	x = np.ones(N)
	fit = fitfunction(x)

	happy = False
	sigma = 1.			# step size / mutation strength
	t = 0

	while not happy:

		t+=1
		happy = termination(t,fit)

		x_ = x + sigma * np.random.randn(N)
		fit_ = fitfunction(x_)

		if fit_ <= fit:
			x = x_
			fit = fit_
			sigma*=np.exp(4/5)
		else: 
			sigma*=np.exp(-1/5)

		print(t,":",fit,sigma,len(x))

	return t


# (1,lambda)-ES / (mu/mu, lambda) with self-adaptation

def sa():

	x = np.ones(N)
	fit = fitfunction(x)
	happy = False
	sigma = 1.	
	t = 0

	tau = 1/np.sqrt(N)
	lambda_ = 100
	mu = 1

	while not happy:

		t+=1
		happy = termination(t,fit)

		xi = tau*np.random.randn(lambda_)
		sigma_ = [sigma * np.exp(xi[i]) for i in range(lambda_)]
		x_= [x + sigma_[i]*np.random.randn(N) for i in range(lambda_)]
		fit_= [fitfunction(x_[i]) for i in range(lambda_) ]

		pop = list(zip(x_,sigma_,fit_))
		pop_ = sorted(pop, key=lambda x: x[2])

		pop_=np.array(pop_)
		x = np.mean(pop_[:mu,[0]],axis=0)[0]
		sigma =np.mean(pop_[:mu,[1]],axis=0)[0]
		fit = np.mean(pop_[:mu,[2]],axis=0)[0]

		print(t,":",fit,sigma)

	return t

# (1,lambda)-ES / (mu/mu, lambda) with derandomized self-adaptation

def derandomized():

	x = np.ones(N)
	fit = fitfunction(x)
	happy = False
	sigma = 1.	
	t = 0

	tau = 1/np.sqrt(N)
	d=1.

	lambda_ = 100
	mu = 1

	while not happy:

		t+=1
		happy = termination(t,fit)

		xi = tau * np.random.randn(lambda_)
		z = [np.random.randn(N) for i in range(lambda_)]
		x_ = [x + np.exp(xi[i]) * sigma * z[i] for i in range(lambda_)]
		sigma_ = [sigma * np.exp(xi[i]/d) for i in range(lambda_)]
		fit_= [fitfunction(x_[i]) for i in range(lambda_) ]


		pop = list(zip(x_,sigma_,fit_))
		pop_ = sorted(pop, key=lambda x: x[2])

		pop_=np.array(pop_)
		x = np.mean(pop_[:mu,[0]],axis=0)[0]
		sigma =np.mean(pop_[:mu,[1]],axis=0)[0]
		fit = np.mean(pop_[:mu,[2]],axis=0)[0]

		print(t,":",fit,sigma)

	return t


# (1,lambda)-ES / (mu/mu, lambda) with evolution path

def evolution_path():

	x = np.ones(N)
	fit = fitfunction(x)
	happy = False
	sigma = 1.	
	t = 0
	c = np.sqrt(1/(N+1))
	d = np.sqrt(1+ np.sqrt(1/N))
	s = np.random.randn(N)

	lambda_ = 10
	mu = 1

	while not happy:

		t+=1
		happy = termination(t,fit)

		z = [np.random.randn(N) for i in range(lambda_)]
		x_= [x + sigma * z[i] for i in range(lambda_)]
		fit_= [fitfunction(x_[i]) for i in range(lambda_) ]

		pop = list(zip(z,x_,fit_))
		pop_ = sorted(pop, key=lambda x: x[2])
		pop_=np.array(pop_)
		z__ =np.mean(pop_[:mu,[0]],axis=0)[0]
		x = np.mean(pop_[:mu,[1]],axis=0)[0]
		fit = np.mean(pop_[:mu,[2]],axis=0)[0]

		s = (1-c)*s + c*z__
		sigma *= np.exp((np.linalg.norm(s)**2/N -1)/d)  

		print(t,":",fit,sigma)

	return t


#results = [oneplusone(),sa(),derandomized(),evolution_path()]
results = [derandomized()]
print(results)