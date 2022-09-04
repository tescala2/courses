import math

#######################
# The bisection method
#######################

def bisect(f,a,b,maxiter):

	FA = f(a)
	FB = f(b)

	if FA*FB >= 0:
		print('ERROR! f(a) and f(b) should have different signs')

	for i in range(0,maxiter):

		p = (a+b)/2
		FP = f(p)

		if FP*FA<0:
			b = p 
		else:
			a = p
			FA = FP

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
p = bisect(f,a,b,maxiter)

print('solution: p=', p)