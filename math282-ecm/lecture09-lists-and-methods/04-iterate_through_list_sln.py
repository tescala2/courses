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


##################
# Problem 1: Write a function that takes as input a list of integers,
# and returns as output the number of odd integers that are in this list. 
##################	

def count_odd_numbers(alist):
	num_occ =0
	for num in alist:
		if num%2 != 0:
			num_occ = num_occ + 1

	return num_occ

my_list = [1,1,2,6,9,7,12,12,7]
print(count_odd_numbers(my_list)) 

##################
# Problem 2: Write a function that takes as input a list of integers,
# and returns as output the same list but in which the even integers
# have been deleted. 
##################	

def delete_even_numbers(alist):
	new_list = []
	for num in alist:
		if num%2 != 0:
			new_list.append(num)

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






