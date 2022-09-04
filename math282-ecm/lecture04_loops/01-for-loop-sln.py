"""
Exercise 1: write a function whose intput is the integer n and whose ouput is the integer 1*2 + 2*3 + 3*4 + ... + n*(n+1) 
"""
def series(n):
	partial_sum = 0
	for i in range(1,n+1):
		partial_sum = partial_sum + i*(i+1)
    return partial_sum

S = series(3)
print(S)


"""
Exercise2: 
Create a function mirror that takes as input a string and returns it in opposit order.
Remark: there is no printing here!
Remark2: you will need to concatenate letters one after the other
to make things easier, start your inverted string with a blank.
"""

def mirror(sen):
	"""
	This function take as input a string
	and return it in reverse order
	"""
	L = len(sen)
	inv_sen = " "
	for i in range(0,L):
		inv_sen = inv_sen + sen[L-1-i]
	return inv_sen

s1 = "my name is Thomas"
s2 = mirror(s1)
print(s2)


"""
 Exercise 7: write a function that takes a string as input and returns a boolean as output. This boolean should be equal to True if the string contains the character "!". Give a meaningful name to your function. Then try it on the sentences: "Hi! My name is Thomas." and "Hi, my name is Thom
"""

def contains_exclamation(sen):
	contain = False
	L = len(sen)
	for i in range(0,L):
		if sen[i] == "!":
			contain = True
	return contain

sen1 = "Hi! My name is Thomas."
sen2 = "Hi, my name is Thomas."

b = contains_exclamation(sen1)
print(b)


