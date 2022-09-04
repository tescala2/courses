import math

#######################
# Newton's method
#######################

def newton(f,df,x_init,maxiter):
	"""
	Newton's method to solve f(x)= 0 on the interval [a,b] 

	INPUTS
    f (function):  function for which we want to solve f(x)=0
    df (function): derivative of f
	x_init (floats): initial guess
	maxiter (int): number of iterations

	OUTPUTS
	x (float): approximate solution of f(x)=0  
	"""
	x = x_init

	for i in range(0,maxiter):
		x = x - f(x)/df(x)
		print(x)

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
N = 10
x = newton(f,df,x0,N)

print('solution: x=', x)