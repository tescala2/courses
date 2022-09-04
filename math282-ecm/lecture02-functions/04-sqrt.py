#####################
"""
Here we first define 2 functions then we are going to use them
"""

def sqrt(value):
    """
    Return the square root of the input value
    """
    output = value**0.5
    return output


def product(value1, value2, value3):
    """
    Returns the product of the three input values.
    """
    prod = value1 * value2
    prod = prod * value3
    return prod


print( sqrt(100) + 5 )
print( product(2,3,10) )


###############################

"""
The name of the argument do not need to match the name chosen
"""

def sqrt(value):
    """
    Return the square root of the input value
    """
    output = value**0.5
    return output


def product(value1, value2, value3):
    """
    Returns the product of the three input values.
    """
    prod = value1 * value2
    prod = prod * value3
    return prod


x = 9
print( sqrt(x) )


y = sqrt(x) + 7
print(y)

z = product(x,y,2)
print(z)


#############################
"""
print is also a function
"""

print( type(product))
print(type(print))


##############
# exercise
"""
write a python function for the polynomial

P(x) = 5x^2 - 10x + 1 

Then use this function to copute P(1) and P(10). 
"""








