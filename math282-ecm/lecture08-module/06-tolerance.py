import math

def newton(f,df,x_init,TOL,verbose):
	"""
	Newton's method to solve f(x)= 0 on the interval [a,b] 

	INPUTS
    f (function):  function for which we want to solve f(x)=0
    df (function): derivative of f
	x_init (floats): initial guess
	TOL (float): tolerance
	verbose (bool): if True, print info at each iterations

	OUTPUTS
	x (float): approximate solution of f(x)=0  
	"""
	x = x_init
	x_old = 1000
	i = 0

	while abs(x-x_old)>TOL:
		x_old = x
		x = x - f(x)/df(x)
		i = i + 1

		if verbose:
			print("iteration", i, "\t x=", x, "\t dx=",abs(x-x_old))

	return x


#####################################
# The function f(x) that we will use
###################################

def f(x):
	y = 3*x - math.exp(x)
	return y

def df(x):
	y = 3 - math.exp(x)
	return y

#######################################################
# Solve f(x)=0 with Newton's method 
#######################################################

x0 = 2
TOL = 1e-4
verbose = True
x = newton(f,df,x0,TOL, verbose)

print('solution: x=', x)