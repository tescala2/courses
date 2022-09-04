"""
Exercise 1: Find the output of these codes without running them (in other words: what would get printed on screen if you were to run these codes?)
"""
# code 1
x0 = 0
x1 = 4
for i in range(0,2):
	x0 = x1
	x1 = (x0+x1)/2
print(x0,x1)

# code 2
x0 = 0
x1 = 4
for i in range(0,2):
	x1 = (x0+x1)/2
	x0 = x1
print(x0,x1)

# code 3
x0 = 0
x1 = 4
for i in range(0,2):
	temp = x1
	x1 = (x0+x1)/2
	x0 = temp
print(x0,x1)

# code 4
x0 = 0
x1 = 4
for i in range(0,2):
	temp = x0
	x0 = x1
	x1 = (x0+x1)/2
print(x0,x1)

# code 5
x0 = 0
x1 = 4
for i in range(0,2):
	temp = x0
	x0 = x1
	x1 = (temp+x1)/2
print(x0,x1)

"""
Exercise 2: Find the output of these codes without running them (in other words: what would get printed on screen if you were to run these codes?)
"""

# code 1
x0 = 0
x1 = 4
for i in range(0,2):
	dist = x1 - x0
	x0 = x1
	x1 = x1 + dist/2
print(x0,x1)

# code 2
x0 = 0
x1 = 4
for i in range(0,2):
	x0 = x1
	dist = x1 - x0
	x1 = x1 + dist/2
print(x0,x1)

# code 3
x0 = 0
x1 = 4
for i in range(0,2):
	dist = x1 - x0
	x1 = x1 + dist/2
	x0 = x1
print(x0,x1)


"""
Exercise 3: write a function whose intput is the integer n and whose ouput is the integer 1^2 + 2^2 + 3^2 + ... + n^2 
"""

"""
Exercise 4: write a function whose intput is the integer n and whose ouput is the n^th number of the Fibonacci sequence (google it).
"""

"""
Exercise 5: what is the largest integer in the the Fibonacci sequence which is smaller than 10 million? Do *not* use the function from problem 4!
"""

"""
Exercise 6: Compute the sum and the product of all the integers that are smaller than 10,000 and which are either a multiple of 7 or a multiple of 15.
"""

"""
Exercise 7: Find the output of these codes without running them (in other words: what would get printed on screen if you were to run these codes?)
"""
# code 1
a = 2
b = 10
x = 0
for i in range(0,20):
	x = x+1
	if x>a and x<b:
		x = x+i
	print(x)


# code 2
a = 2
b = 10
x = 0
for i in range(0,20):
	x = x+1
	if x>a and x<b:
		x = x+i
	    print(x)


# code 3
a = 2
b = 10
x = 0
for i in range(0,20):
	x = x+1
	if x>a and x<b:
		x = x+i
print(x)


"""
Exercise 8: Write a function that takes a string as input and returns an integer as output. This integer should be equal to the number of times the letter "a" appears in this string. Use this function to count the number of times the letter a appear in this paragraph (that is, the text of exercise 8). You will need to use a local variable count, and update it by doing count = count+1.
"""


"""
Exercise 9: Find the output of these codes without running them (in other words: what would get printed on screen if you were to run these codes?)
"""
# code 1
def fun(word):
	L = len(word)
	x = "_"
	for i in range(0,L):
		x = x+"+"
		if word[i] == 'a' or word[i] == 'b':
			x = x+"*"
		else:
			word = word+str(i)
	word = word+x
	return word

w = "boat"
w = fun(w)
print(w)


# code 2
def fun(word):
	L = len(word)
	x = "_"
	for i in range(0,L):
		x = x+"+"
		if word[i] == 'a' or word[i] == 'b':
			x = "_"
		else:
			word = word+"a"
	word = word+x
	return word

w = "boat"
w = fun(w)
print(w)

# code 3
def fun(word):
	L = len(word)
	count = 0
	for i in range(0,L):
		count = count+1
		if word[i] == 'a' or word[i] == 'b':
			count = 0
	return count

w = "boats"
n = fun(w)
print(n)

# code 4
def fun(word):
	L = len(word)
	count = 0
	for i in range(0,L):
		if word[i] == 'a' or word[i] == 'b':
			count = 0
			count = count+1
	return count

w = "boats"
n = fun(w)
print(n)

# code 5
def fun(word):
	L = len(word)
	count = 0
	for i in range(0,L):
		if word[i] == 'a' or word[i] == 'b':
			count = 0
	    count = count+1
	return count

w = "boats"
n = fun(w)
print(n)

"""
Exercise 10: Python tutor exercise.
a) What variable are being created (also gives type and value) as the 5th line of this code (that is, the line started with "for") is  executed for the first time?   
b) What about when it is executed for the second time?
c) What variable are being created (also gives type and value) as the 7th line of this code (that is, the line y=f(x)) is  executed for the second time?
"""
def f(x):
	y = x**2-1
	return y
dx = 0.1
for j in range(0,10):
	x = j*dx
	y = f(x)
	print(y)








