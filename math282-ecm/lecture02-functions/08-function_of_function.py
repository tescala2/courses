"""
Functions are object like any others
"""

n = 10
word = "hello"

def poly(x):
	y = x**2 + 3*x +1
	return y

print(type(n))
print(type(word))
print(type(poly))


# We can manipulate function as the other type:

m = n
w = word
p = poly
print(m)
print(w)

y = poly(2)
print(y)

z = p(2)
print(z)

# write a function that compute the derivative of a mathematical function:

def approximate_derivative(f,x):
	h = 0.01
	derivative = (f(x+h)-f(x))/h
	return derivative

def poly1(x):
	y = x**2 + 3*x +1
	return y

def poly2(x):
	y = 5*x**2 + 3*x +1
	return y

x0 = 10

der1 = approximate_derivative(poly1,x0)
der2 = approximate_derivative(poly2,x0)

print('derivative of  x^2+3x+1 @ 10 =', der1)
print('derivative of 5x^2+3x+1 @ 10 =', der2)










