def greet(name,friend):
	if friend:
		print('Hi', name,'!')
		print('I am so happy to see you!!!')

	if not friend:
        print('Hey', name)
        print('Get out of my way!')


greet('Thomas', False)


#########################
"""
Here is a faster way to do this
If the boolean variable is True then we do the "if block"
If it is false we do the "else block"
"""

def greet(name,friend):
	if friend:
		print('Hi', name,'!')
		print('I am so happy to see you!!!')
	else:
        print('Hey', name)
        print('Get out of my way!')


greet('Thomas', False)

#########################################

def potential_customer(age,state):
	"""
	This function decides wether or not to send a commercial
	about surfboard to a given customer
	"""

	if (state=='CA' or state=='FL') and (age > 10 and age<70):
		output = True
	else:
		output = False

	return output

send = potential_customer(18,'CA')
print(send)


###################
# modulus operator

print(7/3)
print(17//3)

# the remainder of 17//3 is 2 because 17 = 3*5+2
# the modulus operator return the remainder of the integer division
print(17%3)

###########
# example: convert 934 minutes to the format x hours and y minutes 
x = 934//60
y = 934%60
print( '934 minutes =', x, 'hours and', y, 'minutes')

####################
def is_even(n):
	"""
	This function returns true if n is even
	"""
	b = n%2 == 0
	if b:
		output = True
	else:
		output = False
	return output

print(is_even(102))

###############
"""
exercise 1: Write a function that print either 
"This number is a multiple of 5" 
or "This number is NOT a multiple of 5" 
depending on wether or not the input number is a multiple of 5.

"""


#################
# exercise 2: what is the output of this code?

def test(x,y,z):
	if not (x>y and x>z):
		x=x+1
	else:
		x=x-1
	return x

x = 10
y = 1
z = 5

t = test(x,y,z)
print(t)

# exercise 3: what is the output of this code?

def test(x,y,z):
	if not x>y or x>z:
		x=x+1
	else:
		x=x-1
	return x

x = 10
y = 1
z = 5

t = test(x,y,z)
print(t)






