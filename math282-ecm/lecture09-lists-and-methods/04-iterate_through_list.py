# run these code in python tutor!

##############################
# Two ways to loop through a list
##############################

# the regular way

mylist = ["whale", "elephant", "hippo"]
L = len(mylist)

for i in range(0,L):
	print(mylist[i])

# the pythonic way:

mylist = ["whale", "elephant", "hippo"]

for animal in mylist:
	print(animal)

##################
# Another example
##################

# regular way:

mylist = [2,5,3,10]
L = len(mylist)

list_sq = []

for i in range(0,L):
	num = mylist[i]
	list_sq.append( num**2 )

# pythonic way

mylist = [2,5,3,10]
list_sq = []

for num in mylist:
	list_sq.append( num**2 )


#########################################
# Sometime we can not do it the pythonic way
########################################

def dot_prod(v,w):
	n = len(v)
	m = len(w)
	if n != m:
		print("ERROR: v and w should have same length!")

	s = 0
	for i in range(0,n):
		s = s + v[i]*w[i]

	return s

vec1 = [1,2,2]
vec2 = [0,2,3]

print( dot_prod(vec1,vec2))


# remark: the pythonic way is 
# for val1,val2 in zip(v,w):

##################
# Problem 1: Write a function that takes as input a list of integers,
# and returns as output the number of odd integers that are in this list. 
##################	

def count_odd_numbers(alist):

	# COMPLETE

	return num_occ

my_list = [1,1,2,6,9,7,12,12,7]
print(count_odd_numbers(my_list)) 

##################
# Problem 2: Write a function that takes as input a list of integers,
# and returns as output the same list but in which the even integers
# have been deleted. 
##################	

def delete_even_numbers(alist):

	# COMPLETE

	return new_list

list1 = [1,1,2,6,9,7,12,12,7]
list2 = delete_even_numbers(list1)
print(list1) 
print(list2) 


#################
# Nested loop
#################

regular = ["apple", "banana", "pear"]
tropics = ["mango", "pear", "coconut", "pineaple"]
berries = ["raspberry", "strawberry", "blueberry"]

def has_common_item(list1,list2):

	b = False
	for fruit1 in list1:
		for fruit2 in list2:
			if fruit1 == fruit2:
				b = True

	return b

print(has_common_item(regular,tropics))
print(has_common_item(regular,berries))

# let's look at the following nested loop in python tutor:

regular = ["apple",  "pear", "banana",]
tropics = ["mango", "coconut", "pear", "pineaple"]

for fruit1 in regular:
	for fruit2 in tropics:
		if fruit1 == fruit2:
			print("match")
		else:
			print("no match")


#################
# list of list!
#################

"""
consider the matrix 

    | 1 2 3 |
A = | 4 5 6 |
    | 7 8 9 |

we can think of it as a list of list:
"""

row0 = [1,2,3]
row1 = [4,5,6]
row2 = [7,8,9]

matrix = [row0, row1, row2]
print(matrix)

print(matrix[0])
print(matrix[1])
print(matrix[2])

# Now suppose we want to compute the sum of all the square of the entries in the matrix, then take the square root (this is called the Frobenius norm)

s=0
for row in matrix:
	for entry in row:
		s = s + entry**2

print( "Frobenius norm =", s**.5)












