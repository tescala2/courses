###############################
"""
1 - def + name + arg + column
2 - intendation with 4 spaces
3 - docstring 
4 - function is over when intendation ends
5 - Note how this function has no return statement
"""

##################################


def greeting(name):
	"""
	print some greeting
	"""
	string1 = "Hello "
	string2 = ". Nice to meet you."
	sentence = string1 + name + string2
	print(sentence)

greeting("Bob")
greeting("Elizabeth")
greeting("Thomas")


##################################
"""
What is the mistake here?
"""

def greeting(name):
	"""
	print some greeting
	"""
	string1 = "Hello "
	string2 = ". Nice to meet you."
	sentence = string1 + name + string2
	print(sentence)

greeting(Bob)



##################################
"""
This function does not have a return statement!
"""

def greeting(name):
	"""
	print some greeting
	"""
	string1 = "Hello "
	string2 = ". Nice to meet you."
	sentence = string1 + name + string2
	print(sentence)

A = greeting("Bob")
print(A)

#########################
"""
Exercise: write a greeting function that returns a string but does NOT print anything.
So that if you do:

sentence = greeting("Bob")
print(sentence)

it will output:

"Hello Bob. Nice to meet you."
"""


################
# a function is an object (like int, float or str)

x = 2
text = "hello"

def greeting(name):
    """
    print some greeting
    """
    string1 = "Hello "
    string2 = ". Nice to meet you."
    sentence = string1 + name + string2
    print(sentence)


print(x)
print(text)
print(greeting)

print(type(x))
print(type(text))
print(type(greeting))







