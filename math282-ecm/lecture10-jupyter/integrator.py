def Riemann_sum(f,a,b,N):
    """
    Approximate the integral of f(x) on the interval [a,b]
    using a left Riemann sum with step size dx = (b-a)/N:

    INPUT: f (function): function to be integrate
           a,b (floats): left and right boundaries of the interval [a,b]
           N (int): number of cells in which [a,b] is divided

    OUTPUT: integral (float): approximation of the integral of f(x) on [a,b]    

    """

    dx = (b-a)/N
    integral = 0

    for i in range(0,N):
        x = a+i*dx
        integral = integral + f(x)*dx

    return integral