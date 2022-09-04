


###################################################
"""
1 - Note how the input and output variable have different names
2 - r and P are global variable
3 - radius, pi and perimeter are local variable
    They get created when we call the function.
    They are destroyed after the return statement
"""

def circle_perimeter(radius):
    """
    compute the perimeter of a circle
    """
    pi = 3.14159265358979323846
    perimeter = 2*pi*radius
    return perimeter


r = 4
P = circle_perimeter(r)
print(r,P)


#####################################
"""
Let's do an experiment
"""

def circle_perimeter(radius):
	"""
	compute the perimeter of a circle
	"""
    pi = 3.14159265358979323846
    perimeter = 2*pi*radius
    return perimeter


r = 4
P = circle_perimeter(r)
print(r,P)
print(radius, pi, perimeter)

#####################################
"""
This work because the variables radius, pi, and perimeter exists
inside the function.
"""


def circle_perimeter(radius):
	"""
	compute the perimeter of a circle
	"""
    pi = 3.14159265358979323846
    perimeter = 2*pi*radius
    print(radius, pi, perimeter)
    return perimeter


r = 4
P = circle_perimeter(r)
print(r,P)

#####################################
"""
There is a global pi and a local pi. It is OK.
"""

pi = 100


def circle_perimeter(radius):
	"""
	compute the perimeter of a circle
	"""
    pi = 3.14159265358979323846
    perimeter = 2*pi*radius
    print("pi inside ", pi)
    return perimeter


r = 4
P = circle_perimeter(r)
print(r,P)
print("pi outside ", pi)

################
"""
EXERCISE 1:
Write a function that converts Fahrenheit to Celcius. Recall the formula:
temp_c = (temp_f - offset) * multiplier
where offset is 32 and multiplier is 5/9.
Then call this function twice to convert 70 and 90 degree Fahrenheit to Celcius.

Instructions: 
- Give a meaningful name to your function
- The argument of the function should be called temp_f
- Let T1_f = 70 and T2_f = 90 be the variables containing 70 and 90
- Call T1_c and T2_c the two outputs

a) How many local variables do you have in this function? How many global variables
b) Print all this local variables from inside the function (since the function is called twice the local variables should appear twice)

EXERCISE 2:

The grading scheme is:
hwk: 10%
mid1: 20%
mid2: 20%
final: 50%

Student A got the following grades: 98, 91, 85, 88
Student B got 75, 68, 78, 75

Write a function to compute the final grade and call it twice to compute the grade of both students.

"""




