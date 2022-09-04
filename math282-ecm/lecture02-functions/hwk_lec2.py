# Exercise 1: Find the output of this script without running it.
# Then use http://pythontutor.com/ to check your reasonning is correct

constant = 3.14

def mult_by_const(input):
	constant = 2.71
	out = input*constant
	return out

x = 2
y = mult_by_const(x)
print(y)
print(constant)

# Execise 2: Same as above but with the following script:

constant = 3.14

def mult_by_const(input,constant):
	out = input*constant
	constant = 2.71
	print(constant)
	return out

x = 2
y = mult_by_const(x,constant)
print(y)
print(constant)

# Execise 3: Same as above but with the following script:

def mix(word1,word2):
	w = word1 + word2
	print(w)
	w = w + w
	print(w)
	return w

w = 'AB'
v = 'CD'
print(w)
output = mix(w,v)
w = output + '1'
print(w)


# Execise 4: Same as above but with the following script:

x = 2
y = 5

def f(x):
	y = 10
	z = y + x
	print(x)
	return z

print(x)
x = f(y)
print(x) 

# Execise 5: Same as above but with the following script:

def f(A,B):
	print(A)
	A = A+1
	B = B+1
	C = A*B
	print(A)
	return C

C = 3
print(C)
C = f(C,C)
print(C) 

"""
Exercise 6: do all the exercises of section 3.14 of the text.
"""

"""
Exercise 7: Write a function that approximate the integral of any function f(x) between 0 and 1 using a Left Riemann sum with step size 0.25 (go on Wikipedia if you don't remember what is a left Riemann Sum).

Then use this function to approximate the integral of these 3 functions:
f(x) = 5
g(x) = 3x-1
h(x) = x^2 + 2x +1

Of these 3 approximations, which one gave you the exact value?

"""

"""
Exercise 8: Same as the previous problem, but this time using the so called "midpoint rule" when doing the Riemann Sum.
"""

"""
Exercise 9: Write a function that take as arguments two functions f and g as well as a float x and return the product f(x)*g(x). Try your function with:
f(x) = 3x-1
g(x) = x^2 + 2x +1
"""

"""
Exercise 10: Write a function that takes as arguments two functions f and g as well as a float x and return the composition f(g(x)). Try your function with:
f(x) = 3x-1
g(x) = x^2 + 2x +1
"""
