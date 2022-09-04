import math
import solver 

def f(x):
	y = 3*x - math.exp(x)
	return y

def df(x):
	y = 3 - math.exp(x)
	return y

print("================")
print("Newton's method")
verbose = True
x0 = 2
N = 10
solver.newton(f,df,x0,N,verbose)

print("================")
print("Secant method")
verbose = True
x0 = 2
x1 = 3
N = 10
solver.secant(f,x0,x1,N,verbose)

print("================")
print("Bisection method")
verbose = True
a = 1
b = 2
N = 18
solver.bisect(f,a,b,N,verbose)
