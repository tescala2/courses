################################
# 1: A FUNCTION ALLOW YOU TO REUSE THE SAME PIECE OF CODE.

def triangle_area(base,height):
    """
    this function compute the area of a triangle
    """
    area = base*height/2
    return area


A1 = triangle_area(5,4)
print(A1)

A2 = triangle_area(3,3)
print(A2)

A3 = triangle_area(10,2.2)
print(A3)


################################
# 2: PRINT RETURN VALUE WITHOUT SAVING IT

def triangle_area(base,height):
    """
    this function compute the area of a triangle
    """
    area = base*height/2
    return area

print( triangle_area(5,4) )
print( triangle_area(3,3) )
print( triangle_area(10,2.2) )



################################
# 3: SOME TYPICAL ERROR MESSAGES

def triangle_area(base,height):
    """
    this function compute the area of a triangle
    """
    area = base*height/2
    return area

print( triangle_area(5) )
print( triangle_area() )
print( triangle_area(2,2,2) )

###################
# 4: exercise:

"""
Write a function that computes the volume of a cone (google it)
Then compute the volume of:
-- a cone of radius 2 and height 10
-- a cone of radius 10 and height 2
"""




