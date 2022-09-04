# Find the output without running the code

x=1

def test():
	x=2
	print(x)

test()
print(x)

############################

x=1

def test():
	y = 2
	print(y)

test()
print(y)

############################

x=1

def test():
	y = 2
	print(y)
	print(x)

test()

############################

"""
LOCAL VARIABLE CAN NOT BE ACCESSED OUTSIDE OF THE FUNCTION
GLOBAL VARIABLE CAN BE ACCESSED EVERYWHERE (INCLUDING IN THE FUNCTION)
Some people think it is bad practice to access global variables inside a function
"""

"""
Consider the 2 codes belows that compute the average value of a function. Both are correct, both uses global variables inside functions. One is better written than the other
"""

import math

def f(x):
	y = math.exp(-x**2)
	return y

def Riemann_sum(f,a,b,N):
	"""
	Approximate the integral of f(x) on the interval [a,b]
	using a left Riemann sum with step size dx = (b-a)/N:
	"""
	dx = (b-a)/N
	integral = 0
	for i in range(0,N):
		x = a+i*dx
		integral = integral + f(x)*dx
	return integral


def function_average(f,a,b,N):
	integral = Riemann_sum(f,a,b,N)
	mean_val = integral/(b-a)
	return mean_val


a = 0
b = 1
N = 1000
print(function_average(f,a,b,N))

####################################
####################################

import math

def f(x):
	y = math.exp(-x**2)
	return y

def Riemann_sum(a,b,N):
	"""
	Approximate the integral of f(x) on the interval [a,b]
	using a left Riemann sum with step size dx = (b-a)/N:
	"""
	dx = (b-a)/N
	integral = 0
	for i in range(0,N):
		x = a+i*dx
		integral = integral + f(x)*dx
	return integral


def function_average(a,b,N):
	integral = Riemann_sum(a,b,N)
	mean_val = integral/(b-a)
	return mean_val


a = 0
b = 1
N = 1000
print(function_average(a,b,N))




