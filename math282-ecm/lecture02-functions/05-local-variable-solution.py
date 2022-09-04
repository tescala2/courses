def convert_to_celcius(temp_f):
    """
    input: temp in Fahrenheit
    output: temp in Celcius
    """
    offset = 32
    multiplier = 5/9
    temp_c = (temp_f - offset) * multiplier
    return temp_c

T1_f = 70
T2_f = 90

T1_c = convert_to_celcius(T1_f)
T2_c = convert_to_celcius(T2_f)

print(T1_c)
print(T2_c)

"""
There are 4 local variables: 
offset
multiplier
temp_c
temp_f

There are 5 global variables:
T1_f
T2_f
T1_c
T2_c
convert_to_celcius
"""


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

