# Exercise 1:
"""
Write a function greet(name, loud) where name is a string and loud is a boolean. The function print 
"HELLO Thomas!!!"
if loud is True (here I am assuming that the name is Thomas) or
"Hello Thomas"
if loud is False.

"""
def greet(name,loud):
	#complete

##############################################
##############################################

# Exercise 2a: Find the output without running the code:
x = True
y = True
b = not( x or not y)
print(b)

# Exercise 2b: Find the output without running the code:
x = True
y = False
b = not( x or not y)
print(b)

# Exercise 2c: Find the output without running the code:
x = False
y = True
b = not( x or not y)
print(b)


# Exercise 2d: Find the output without running the code:
x = False
y = False
b = not( x or not y)
print(b)

##############################################
##############################################

# Exercise 3a: Find the output without running the code:
x = True
y = False
b = not y
print(b)

# Exercise 3b: Find the output without running the code:
x = True
y = False
b = y==True
print(b)

# Exercise 3c: Find the output without running the code:
x = True
y = False
b = not y == x
print(b)

# Exercise 3d: Find the output without running the code:
x = True
y = False
b = y!=True
print(b)

##############################################
##############################################

"""
Exercise 4: 
Two expressions are logically equivalent if they have the same value for all possible values of the variables x and y.
For each of the expression below, find a logically equivalent expression which is simpler.
# Example (x>y) or (x==y) can be simplified to x>=y
# Example: not x<=y can be simplified to  x>y
"""

(x>=y) and (x==y)
not( x<y )
not (x==y or x<y)
(x>y) or (x==y)
not(x<5 or x>10)
not x<5 or x>10

##############################################
##############################################

"""
Exercise 5
Consider the following function:
"""

def fun(bool1, bool2):
    """
    Take two Boolean values bool1 and bool2
    and return the specified Boolean values
    """
    
    if bool1:
        if bool2:
            x= False
        else:
            x = True
    else:
        x = True

    return x

# Which Boolean expression below is logically equivalent to the above function?

bool1 and bool2
bool1 or bool2
not( bool1 or bool2)
not (bool1 and bool2)


##############################################
##############################################

"""
Exercise 6: Write a function that takes as input the variables name (a string) and age (an int) and print the age of the given person. Here are a few example of how your function should behave:

if you call
age("Bob", 31)
then the function should print:
"Bob is 31 years old"

if you call
age("Bob", -5)
then the function should print:
"ERROR: invalid age!"

if you call
age("Bob", 150)
then the function should print:
"ERROR: Nobody can be that old!"

As you see in the above example, if the age is less than 0, or greater than let say 120, the function should print a special string.
"""

##############################################
##############################################

"""
Exercise 7: write a function solve_quadratic(a,b,c) that print the solutions of 

ax^2 + bx + c = 0

as you know there are three cases, depending on whether or not the determinant is greater, equal, or less than zero.

Example:

If you call:
solve_quadratic(1,0,-4)
then the function should print: 
"The roots are x1=2 and 

If you call:
solve_quadratic(1,-4,4)
then the function should print: 
"There is a double root at x=2

If you call 
solve_quadratic(1,0,4)
then the function should print:
"no real roots"
"""

def solve_quadratic(a,b,c):
	# complete



##############################################
##############################################

#Exercise 8: Do Exercise 2, part 1 in section 5.14 of the text

##############################################
##############################################

#Exercise 9: Do Exercise 3, part 1 in section 5.14 of the text










