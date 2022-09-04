###################################
###################################
# Find the minimum of a function in the interval [0,3] and use stepsize 0.5

def poly(x):
	y = x**2-2*x+5
	return y

current_min_val = 10000
x = 0
dx = 0.5

for i in range(0,7):
	y = poly(x)
	if y < current_min_val:
		current_min_val = y
	x = x + dx

print(current_min_val)

###################################
###################################
# Exercise: Find the minimum of the function 3x^2-50x+1 in the interval [-10,10] and use stepsize 0.001. 

####################################
####################################
# marching through an interval [a,b] using step size dx

a = 5 # left boundary of the interval
b = 7 # right boundary of the interval
N = 100 # number of cells 
dx = (b-a)/N # step size

# this loop start at x=a and finish at x=b
for i in range(0,N+1):
	x = a+i*dx
	print(x)

# equivalently we can do that... (actually, there is a problem in the code below... what is it? How to fix it?)
x = a
for i in range(0,N+1):
	x = x+dx
	print(x)

####################################
####################################
# Riemann Sum

def Riemann_sum(f,a,b,N):
	"""
	Approximate the integral of f(x) on the interval [a,b]
	using a left Riemann sum with step size dx = (b-a)/N:

	INPUT: f (function): function to be integrate
		   a,b (floats): left and right boundaries of the interval [a,b]
		   N (int): number of cells in which [a,b] is divided

    OUTPUT: integral (float): approximation of the integral of f(x) on [a,b]  	

	"""

	dx = (b-a)/N
	integral = 0

	for i in range(0,N):
		x = a+i*dx
		integral = integral + f(x)*dx

	return integral



def f(x):
	y = x**2+1
	return y

a = 1
b = 3
N = 100
print('code:', Riemann_sum(f,a,b,N)) 
print('expected:', (27/3+3) - (1/3+1))







