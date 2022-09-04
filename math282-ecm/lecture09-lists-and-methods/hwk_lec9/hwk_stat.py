"""
PROBLEM 1

Create a library (a "module" in python language) called statbasic.py 
This library should have the following function: 
my_mean() 
my_min() 
my_max()
my_weighted_average()

The first three function take as input a list of floats and return
the mean, min, and max of the floats contained in this list.

The last function takes 2 inputs (a list grades and a list of weights)
and compute the weighted average of these grades. 
"""

import statbasic

grades = [90, 85, 92, 70, 97, 68, 79, 82, 88]

average = # complete
best = # complete
bottom = # complete
print(average, best, bottom)

# In math 282 the grading scheme is:
# quiz 22%, midterm1 22%, midterm2 22%, final 34% 
weight_math282 = [.22, .22, .22, .34]
grades = [100,95,92,93]
overall_grade = # complete
print(overall_grade)

"""
PROBLEM 2

In python it is easy to remove an element from a list with the pop() method.
Try the piece of code below:

grades = [90, 85, 92, 70]
grades.pop(2)
print(grades)

Write a function that returns the top k values contained in a vector. 
Here is an example with k = 3

input: [6,3,9,1,7,2] 
output: [9,7,6]

Note that the output is a list containing the top k values in decreasing order.

"""

def topk(vec,k):

    lst = []

    # complete

    return lst

grades = [90, 85, 92, 70, 97, 68, 79, 82, 88, 56, 96]
print( topk(grades,3))







