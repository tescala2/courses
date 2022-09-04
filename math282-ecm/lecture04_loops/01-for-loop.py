##################
##################
# for loop works as follow:

for i in range(0,4):
	print("hello")


for i in range(5,10):
	print(i)
	print(2*i)
	print('------')



###################################
###################################
# let's look at this in Python tutor.
x=0

for i in range(1,5):
    x = x+i**2
    x = x*2
    x = x-10

x = x+100
x = x*10
print(x)


###################################
###################################
# An other example: compute factorial of n:
n = 4
fac = 1
for i in range(2,n+1):
	fac = fac*i
print(fac)

# be careful about intendation:
n = 4
fac = 1
for i in range(2,n+1):
	fac = fac*i
	print(fac)

###################################
###################################
# Exercise 1: write a function whose intput is the integer n and whose ouput is the integer 1*2 + 2*3 + 3*4 + ... + n*(n+1) 

###################################
###################################
# with string
sen = "My name is thomas"
print(len(sen))
print(sen[0])


###################################
###################################

sen = "My name is thomas"
for i in range(0,17):
	print(sen[i]) 

###################################
###################################
# print in reverse order
sen = "My name is thomas"
for i in range(0,17):
    print(sen[16-i])

"""
Exercise2: 
Create a function mirror that takes as input a string and returns it in opposit order.
Remark: there is no printing here!
Remark2: you will need to concatenate letters one after the other
to make things easier, start your inverted string with a blank.
"""

###################################
###################################
# lexical comparison:
print("b"<"z")
print("c"<="c")

###################################
###################################
# find the largest letter in a string:
sentence = "greater"
print(len(sentence))
current_largest_letter = "a"
for j in range(0,7):
	letter = sentence[j]
	if letter > current_largest_letter:
		current_largest_letter = letter
print(current_largest_letter)

###################################
###################################
# Exercise 3: What is the output of this code
sentence = "this"
current_smallest_letter = "a"
for j in range(0,4):
	letter = sentence[j]
	if letter < current_smallest_letter:
		current_smallest_letter = letter
    print(j,current_smallest_letter)

# Exercise 4: What is the output of this code
sentence = "this"
current_smallest_letter = "z"
for j in range(0,4):
	letter = sentence[j]
	if letter < current_smallest_letter:
		current_smallest_letter = letter
    print(j,current_smallest_letter)

# Exercise 5: What is the output of this code
sentence = "this"
current_smallest_letter = "z"
for j in range(0,4):
	letter = sentence[j]
	if letter < current_smallest_letter:
		current_smallest_letter = letter
        print(j,current_smallest_letter)

# Exercise 6: What is the output of this code
sentence = "this"
current_smallest_letter = "z"
for j in range(0,4):
	letter = sentence[j]
	if letter < current_smallest_letter:
		current_smallest_letter = letter
print(j,current_smallest_letter)

"""
 Exercise 7: write a function that takes a string as input and returns a boolean as output. This boolean should be equal to True if the string contains the character "!". Give a meaningful name to your function. Then try it on the sentences: "Hi! My name is Thomas." and "Hi, my name is Thomas."
 """



