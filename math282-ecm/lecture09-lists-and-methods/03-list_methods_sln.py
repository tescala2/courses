##########################
# A list method: append()
##########################

fruits = ["coconuts","pineaple"]
print(fruits)

fruits.append("apple")
print(fruits)

fruits.append("pear")
print(fruits)

#######################################
# That's why empty list are convenient!
#######################################

greetings = []
print(greetings)

greetings.append("hello!")
print(greetings)

greetings.append("hi!")
print(greetings)

greetings.append("nice day!")
print(greetings)

#####################
# Example with a loop
#####################

mylist = [2,5,3,10]
L = len(mylist)

list_sq = []
for i in range(0,L):
	num = mylist[i]
	list_sq.append( num**2 )

print(mylist)
print(list_sq)

##############
# Exercise 5
##############

# Complete the function below that takes as input a list
# and returns as output the same list but in reverse order

def list_reverse(alist):
	L = len(alist)
	rev_list = []
	for i in range(1,L+1):
		item = alist[-i]
		rev_list.append(item)
	return rev_list

fruits = ["banana", "apple", "orange", "coconuts"]

fruits2 = list_reverse(fruits)
print(fruits)
print(fruits2)

##############################
# The in-place vs not in place
##############################

# the reverse method is in place because it changes the input list
fruits = ["banana", "apple", "orange", "coconuts"]
fruits.reverse()
print(fruits)

# the upper method is NOT in-place
word = "hello"
word.upper()
print(word)

# compare with 
word = "hello"
word2 = word.upper()
print(word)
print(word2)


# the reverse method is in place because it changes the input list
fruits = ["banana", "apple", "orange", "coconuts"]
fruits2 = fruits.reverse()
print(fruits)
print(fruits2)


# the upper() method returns a string and does not modify the original string
# the reverse() and append() methods return nothing, but they do modify the original list

