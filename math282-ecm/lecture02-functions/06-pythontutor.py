###############################
"""
http://pythontutor.com/
"""
# Note how each line is executed one after the other

x0 = 2
x1 = 5
x2 = 10

s = 0
s = s + x0
s = s + x1
s = s + x2

print(s)

###############################
# Note how each line is NOT executed one after the other.

def vec_length(v1,v2,v3):
	"""
	compute the length of a vector in 3D
	"""
	length_sq = v1*v1 + v2*v2 + v3*v3
	length = length_sq**0.5 
	return length

x=0
y=1
z=1
L = vec_length(x,y,z)
print(L)

###############################

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


################################



def grade(hwk,mid1,mid2,final):
	"""
	Compute grade under grading scheme:
	hwk: 10%
	mid1: 20%
	mid2: 20%
	final: 50%
	"""
	hwk_wgh = 0.1
	mid1_wgh = 0.2
	mid2_wgh = 0.2
	final_wgh = 0.5
	overall = hwk_wgh*hwk + mid1_wgh*mid1 + mid2_wgh*mid2 + final_wgh*final
	return overall


# student A
h_A = 98
m1_A = 91
m2_A = 85
f_A =  88

# student B
h_B = 75
m1_B = 68
m2_B = 78
f_B =  75

# compute grades
grade_A = grade(h_A,m1_A,m2_A,f_A)
grade_B = grade(h_B,m1_B,m2_B,f_B)
print(grade_A, grade_B)

############################

def greeting(name):
	"""
	print some greeting
	"""
	string1 = "Hello "
	string2 = ". Nice to meet you."
	sentence = string1 + name + string2
	return sentence

name1 = "Bob"
name2 = "Elizabeth"
greet1 = greeting(name1)
greet2 = greeting(name2)
print(greet1)
print(greet2)






