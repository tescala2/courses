"""
Exercise 1: Complete the function below that detects if a letter is repeated back to back in a string. For example if the input is "abbc", then the output should be True, but if the input is "abcb", then the output should be False.

Here are some more example:
input: "hello" output: True
input "food" output: True
input "invitation" output: False 
input "yesterday" output: False  
"""

def is_repeated(text):
	# complete

print(is_repeated("hello"))
print(is_repeated("food"))
print(is_repeated("invitation"))
print(is_repeated("yesterday"))



"""
Exercice 2: Complete the function below that uses brute force to find ALL the solutions of f(x)=0 on the interval [a,b]. 

This function simply marches through the interval [a,b], using step size dx = (b-a)/N, and detect when f(x) is crossing the x-axis. This function will simply PRINT all the solutions of f(x)=0. Nothing is returned.

Then use your function to solve:
(x-pi/2)(x-pi)=0
cos(x)=0
x^2+1=0
on the interval [0,5]. 
Obviously all these equation are solvable with pen and paper. The goal here is to debug.

Remark1: You are not allowed to use python containers (e.g. list)
Remark2: Make sure your code do only 1 function evaluation per step. 
"""
import math

def print_all_zeros(f,a,b,N):
	"""
	Print all the solutions of f(x)=0 in interval [a,b]

	INPUT: f (function): Function for which we want to solve f(x)=0
		   a,b (floats): Left and right boundaries of the interval [a,b]  
		   N (int): number of cells in which [a,b] is divided.
	OUTPUT: None 
	"""
	
	# complete


def f(x):
	y = (x-math.pi/2)*(x-math.pi)
	return 

def g(x):
	#complete
	return y

def h(x):
	# complete
	return y

a = 0
b = 5
N = 10000
print_all_zeros(f,a,b,N)
print_all_zeros(g,a,b,N)
print_all_zeros(h,a,b,N)




"""
Exercise 3: modify the function from the previous problem so that, if f(x)=0 has no solutions, it prints "no solution"
"""




"""
Exercise 4: Write a function that count the number of solutions to f(x)=0 on the interval [a,b]. Then test your function on:
(x-pi/2)(x-pi)=0
cos(x)=0
x^2+1=0
on the interval [0,4*pi] (This interval is different than in the previous problems!) 
"""
import math

def count_zeros(f,a,b,N):
	"""
	Solve f(x)=0 by simply marching through the interval [a,b]

	INPUT: f (function): Function for which we want to solve f(x)=0
		   a,b (floats): Left and right boundaries of the interval  
		   N (int): number of cells in which [a,b] is divided
	OUTPUT: num_zeros (int): number of solutions to f(x)=0 
	"""

	# complete




"""
Exercice 5: Complete the function below that use brute force to find ALL the local min of f(x) on the interval [a,b]. 

This function simply marches through the interval [a,b], using step size dx = (b-a)/N, and detects when f(x) has a local min, that is, it is looking for x so that f(x) <= f(x-dx) and f(x) <= f(x+dx). 

This function will simply PRINT all the local min. Nothing is returned.
If there are no local mins, the function should print "No local mins".

Use the following functions to debug your code:
f(x) = (1/4)x^4 - 2x^3 + (11/2)x^2 - 6x + 1 
g(x) = cos(x)
h(x)= (x-pi)^2+1
on the interval [0,4pi]. The function f(x) has local min at x=1 and x=3.

Remark1: You are not allowed to use python containers (e.g. list)
Remark2: Make sure your code do only 1 function evaluation per step. 
"""
import math

def print_all_mins(f,a,b,N):
	"""
	Print all the solutions of f(x)=0 in interval [a,b]

	INPUT: f (function): Function for which we want to solve f(x)=0
		   a,b (floats): Left and right boundaries of the interval [a,b]  
		   N (int): number of cells in which [a,b] is divided.
	OUTPUT: None 
	"""
	
	# complete



"""
exercise 6: Consider the function

f(x) = (x-1.2)(x-3.1)(x-3.2)

This function has obviously 3 zeros? Suppose you run the bisection method on this function with initial interval [0,4]. Figure out, without running any code, which of the 3 roots will be chosen by the bisection method.

"""


"""
exercise 7: Same question has above, but with the function

g(x) = (x-2.2)(x-2.9)(x-3.1)

"""



"""
exercise 8: Write a code and use the bisection method to solve

(x-1.2)(x-3.1)(x-3.2) = 0
(x-2.2)(x-2.9)(x-3.1) = 0
2*cos(x/2) = exp(-x)

Use [a,b]=[0,4] as your starting interval. Check that the bisection method indeed chose the zeros that you have predicted in the past 2 problems (for the first 2 equations).

"""


	




