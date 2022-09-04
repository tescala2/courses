import math

#######################
# The bisection method
#######################

def bisect(f,a,b,maxiter):

	for i in range(0,maxiter):

		p = (a+b)/2

		if f(p)*f(a)<0:
			b = p 
		else:
			a = p

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
maxiter = 10
p = bisect(f,a,b,maxiter)

print('solution: p=', p)






