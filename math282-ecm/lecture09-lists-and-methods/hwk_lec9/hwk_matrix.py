"""
PROBLEM 1:

Write a function that compute the norm of a vector. For example if

v = [1,10,10,3]

then ||v|| =  sqrt( 1^2 + 10^2 + 10^2 +3^2 )

Use the pythonic way to do a loop.

"""
import math

def norm(v):
	# complete

v = [1,10,10,3]
print( norm(v))

"""
PROBLEM 2

Write a function that returns True if 2 vectors are perpendicular
and False otherwise.

Also we want to include some tolerence so that if the 2 vectors
are almost perpendicular, it will return true. To be more precise,
if the absolute value of the dot product between the 2 vectors is smaller
than TOL, then we want the function to return True

"""


def is_perp(v,w,TOL):
    """
    Return true if the |v.w|<TOL
    """
    # complete


vec1 = [1,2,0]
vec2 = [-2,1.00003,5]
tol = 1e-3

print(is_perp(vec1,vec2,tol))
print(is_perp(vec1,vec2,0))


"""
PROBLEM 3

Write a function that does matrix vector multiplication.

"""

def mat_vec_mul(A,v):
    """
    Compute matrix A times vector v.
    The matrix A is a list of list.
    For example A = [[1,2,3,1], [1,1,1,1], [0,5,5,2]] 
    corresponds to the matrix

         |1 2 3 1|
    A =  |1 1 1 1|
         |0 5 5 2|

    INPUT: A (list of list of floats) The matrix
           v (list of floats) The vector

    OUTPUT: w (list of float) The result of A*v

    """
    
    # complete

    return w

W = [[1,2], [1,1], [0,3]  ]
vec = [1,5]
print(mat_vec_mul(W,vec))

W = [[1,2,3,1], [1,1,1,1]]
vec = [1,0,1,2]
print(mat_vec_mul(W,vec))


"""
PROBLEM 4

Write a function that compute the transpose of a matrix

For example if A is the matrix [[0,1,2,3,4], [5,6,7,8,9]], that is

    |0 1 2 3 4|
    |5 6 7 8 9|

then its transpose is [[0,5],[1,6],[2,7],[3,8],[4,9]]

    |0 5|
    |1 6|
    |2 7|
    |3 8|
    |4 9|

To do this problem you will start with an empty list B = []
and an empty row r=[]. Then you will fill the row by appending
value to it. Once the row is filled you will append it to the empty 
list B, etc...

You will need to use nested loops.

"""

def transpose(A):

    m = len(A) # height of the matrix
    n = len(A[0]) # width of the matrix
    B = []

    # complete

    return B

A = [[0,1,2,3,4], [5,6,7,8,9]]
B = transpose(A)
print(B)    






