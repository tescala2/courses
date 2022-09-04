import numpy as np
from scipy import linalg


def grad_desc(gradf,alpha,x0,num_iter):
    """
    Gradient descent algorithm
    This function returns ALL the iterates 

    INPUT
    gradf (function): gradient of the function to minimize
    alpha (float): step size
    x0 (1D-array): initial guess
    num_iter (int): number of iteration

    OUTPUT
    iterates (2D-array): matrix containing all the iterates 
    """
    x = x0
    iterates = np.zeros( (num_iter+1,2))
    for i in range(0,num_iter):
        iterates[i,:] = x
        x = x - alpha*gradf(x)
    iterates[-1,:] = x
    return iterates


def newton(gradf,Hf,x0,num_iter):
    """
    Newton's method 
    This function returns ALL the iterates

    INPUT
    gradf (function): gradient of the function to optimize
    Hf (function): Hessian of the function to optimize
    x0 (1D-array): initial guess
    num_iter (int): number of iteration

    OUTPUT
    iterates (2D-array): matrix containing all the iterates 
    """


    x = x0
    iterates = np.zeros( (num_iter+1,2))
    for i in range(0,num_iter):
        iterates[i,:] = x
        y = linalg.solve(Hf(x), - gradf(x))
        x = x + y
    iterates[-1,:] = x
    return iterates


def grad_desc2(gradf,alpha,x0,TOL):

    """
    Gradient descent algorithm. 
    The algorithm stop when the distance between
    two successive iterates is less than TOL 
    This function returns ONLY the last iterate.
    This function should also PRINT the number of iteration that
    where performed. 

    INPUT
    gradf (function): gradient of the function to minimize
    alpha (float): step size
    x0 (1D-array): initial guess
    num_iter (int): number of iteration

    OUTPUT
    iterates (2D-array): matrix containing all the iterates 
    """
    i = 0
    dist_btw_iterates = 100000
    x = x0
    while dist_btw_iterates > TOL:
        i = i+1
        xnew = x - alpha*gradf(x)
        dist_btw_iterates = linalg.norm(xnew-x)
        x = xnew
    print('number of iteration needed to reach desired tolerance: ', i)
    return x


def newton2(gradf,Hf,x0,TOL):

    """
    Newton's algorithm. 
    The algorithm stop when the distance between
    two successive iterates is less than TOL 
    This function returns ONLY the last iterate.
    This function should also PRINT the number of iteration that
    where performed. 

    INPUT
    gradf (function): gradient of the function to minimize
    Hf (float): Hessian of the function to minimized
    x0 (1D-array): initial guess
    num_iter (int): number of iteration

    OUTPUT
    iterates (2D-array): matrix containing all the iterates 
    """
    i = 0
    dist_btw_iterates = 100000
    x = x0
    while dist_btw_iterates > TOL:
        i = i+1
        y = linalg.solve(Hf(x), - gradf(x))
        xnew = x + y
        dist_btw_iterates = linalg.norm(xnew-x)
        x = xnew
    print('number of iteration needed to reach desired tolerance: ', i)
    return x

