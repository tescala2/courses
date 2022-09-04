################################
"""
using print() or return?
"""

def greeting(name):
	"""
	print some greeting
	"""
	string1 = "Hello "
	string2 = ". Nice to meet you."
	sentence = string1 + name + string2
	print(sentence)

name1 = "Bob"
name2 = "Elizabeth"
greeting(name1)
greeting(name2)


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