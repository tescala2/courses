import math

#######################
# The bisection method
#######################

def bisect(f,a,b,maxiter, verbose):
	"""
	Bisection method algorithm .
	Solve f(x)= 0 on the interval [a,b]
	f(a) and f(b) must have different sign 

	INPUTS
    f (function):  function for which we want to solve f(x)=0 
	a, b (floats): boundaries of the search interval.
	maxiter (int): number of iterations
	verbose (bool): if True, print info at each iterations

	OUTPUTS
	p (float): approximate solution of f(x)=0  
	"""

	FA = f(a)
	FB = f(b)

	assert FA*FB < 0 , 'f(a) and f(b) should have different signs'

	for i in range(0,maxiter):

		p = (a+b)/2
		FP = f(p)

		if FP*FA<0:
			b = p 
		else:
			a = p
			FA = FP
		if verbose:
			print('iteration', i, '\t [', a , b, ']')

	p = (a+b)/2

	return p

#####################################
# The function f(x) that we will use
###################################

def f(x):
	y = 3*x - math.exp(x)
	return y

#######################################################
# Solve f(x)=0 with bisection method on interval [a,b]
#######################################################

a = 1
b = 2
maxiter = 20
verb = True
p = bisect(f,a,b,maxiter,verb)

print('solution: p=', p)