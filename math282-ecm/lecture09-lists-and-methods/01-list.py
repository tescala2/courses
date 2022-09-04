
######################################
# list uses square braquets and commas 
######################################

# list of int:
mylist = [3,5,10,0]
print(mylist)

# list of strings
names = ["Thomas", "Elizabeth", "Bob"]
print(names)

# list of boolean
bool_list = [True, True, False]
print(bool_list)

# empty list!
mylist = []
print(mylist)

# mixing types (not recommended!)
lst = ["Hello!", 3.14, True]
print(lst)

# Remark: we have a new type: 
# int, floats, string, boolean, functions and LISTS!
names = ["Thomas", "Elizabeth", "Bob"]
print( type(names) )


#################################
# indexing 
#################################

animals = ["whale", "elephant", "hippo", 
                       "cow", "pig", "cat", "mouse", "mosquito"]

# first item
print( animals[0] )

# third item
print( animals[2] )

# length of the list
num = len(animals)
print(num)

# last item (which is item 7)
print( animals[num-1] )

# or equivalently
print(animals[-1])

# negative indexing
print( animals[-1] )
print( animals[-2] )
print( animals[-3] )
print( animals[-4] )
print( animals[-5] )

# out of bound
print( animals[10] )
print( animals[-10] )

####################
# Changing values
####################

lst = [0,2,4,6,8,10]
print(lst)

lst[2] = -1
print(lst)

animals = ["whale", "elephant", "hippo"]
print(animals)

animals[-1] = "tiger"
print(animals)


#################################
# slicing 
#################################

animals = ["whale", "elephant", "hippo", 
                       "cow", "pig", "cat", "mouse", "mosquito"]

sublist = animals[2:6]
print(sublist)

sublist = animals[1:4]
print(sublist)

sublist = animals[1:2]
print(sublist)

sublist = animals[1:1]
print(sublist)

# open ended slices
num = len(animals)

sublist = animals[4:num-1]
print(sublist)

sublist = animals[4:num]
print(sublist)

sublist = animals[4:]
print(sublist)

sublist = animals[0:4]
print(sublist)

sublist = animals[:4]
print(sublist)

# using negative indices
sublist = animals[-2:]
print(sublist)

sublist = animals[-3:]
print(sublist)

sublist = animals[0:-1]
print(sublist)

# what if we go overboard
sublist = animals[4:]
print(sublist)

sublist = animals[4:1000]
print(sublist)

# empty list
sublist = animals[100:200]
print(sublist)

sublist = animals[4:2]
print(sublist)



""" 
We have seen 6 types of objects:
1) int, 2)floats, 3)string, 4)boolean, 5)functions, 6)lists

Strings and lists are called sequence type. 
Both of them can be indexed and sliced
"""

sentence = "Hello, nice to meet you"

# len() function
num_letters = len(sentence)
print(num_letters)

# indexing
print( sentence[0] )
print( sentence[1] )
print( sentence[2] )

# negative indexing
print( sentence[-1] )
print( sentence[-2] )

# slicing
print( sentence[7:11])
print( sentence[:11])
print( sentence[11:])
print( sentence[-3:] )


###############
# Loops
################

mylist = [1,10,5,3]
L = len(mylist)

for i in range(0,L):
	print( mylist[i]**2 )



# Problem 1: Make a list called vowels that contains the 6 vowels of the alphabet.
# a) print the third item (index=2 due to zero indexing) of the list
# b) print the first 3 items by slicing the list
# c) print items with indices 2,3,4 by slicing the list
# d) make a loop to print:
# 	vowel 1 is a
# 	vowel 2 is e
#	vowel 3 is i
# 	etc...



#########################################################
# the split() method convert a string to a list of string
#########################################################

text = "Machine learning is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions , relying on patterns and inference instead . It is seen as a subset of artificial intelligence"

mylist = text.split()
print(mylist)


# Problem 2. How many words in this text? How many letters in this text?



# Problem 3. Create a list that contains the 10 words which are in the middle of the text.


# Problem 4. Create two lists called sentence1 and sentence2 that contains respectively the words in the first sentence and the words of the second sentence. Use negative indexing.















