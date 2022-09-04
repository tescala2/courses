"""
This module contains 3 algorithms to solve f(x)=0
"""

def newton(f,df,x_init,maxiter,verbose):
	"""
	Newton's method to solve f(x)= 0 on the interval

	INPUTS
    f (function):  function for which we want to solve f(x)=0
    df (function): derivative of f
	x_init (floats): initial guess
	maxiter (int): number of iterations
	verbose (bool): if True, print info at each iterations

	OUTPUTS
	x (float): approximate solution of f(x)=0  
	"""
	x = x_init

	for i in range(0,maxiter):
		x = x - f(x)/df(x)
		if verbose:
			print("iteration", i, "\t x=", x)

	return x


def secant(f,x_init_1, x_init_2, maxiter, verbose):
	"""
	Secant method to solve f(x)= 0 

	INPUTS
    f (function):  function for which we want to solve f(x)=0
	x_init_1, x_init_2 (floats): initial guesses
	maxiter (int): number of iterations
	verbose (bool): if True, print info at each iterations

	OUTPUTS
	x (float): approximate solution of f(x)=0  
	"""
	x_old = x_init_1
	F_old = f(x_old)
	x_current = x_init_2

	for i in range(0,maxiter):
		F_current = f(x_current)
		der = (F_current - F_old)/(x_current-x_old)
		x_new = x_current - F_current/der
		F_old = F_current
		x_old = x_current
		x_current = x_new
		if verbose:
			print("iteration", i, "\t x=", x_current)

	return x_current


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
			print('iteration', i, '\t x=', (a+b)/2)

	p = (a+b)/2

	return p