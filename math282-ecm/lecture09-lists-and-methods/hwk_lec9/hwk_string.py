"""
The + and * operators are quite convenient when dealing with lists.
You will need them in this hwk.
Try the lines below to understand how they work
"""
my_list = [1,5] + [10,2]
my_list = [1,5] + [1,5]
my_list = [1,5] * 3
my_list = [1,5] * 10
my_list = [0] * 5
my_list = [0] * 10
my_list = [7] * 10


"""
PROBLEM 1

Write a function that compute the number of times each word in a given list of words appears in a text. 
You are not allowed to use the count() built in method. 
Also you will need to initialize a list filled with zeros as explained above.
Read the docstring below and complete the function.
"""

def count_words(text, wd_list):
    """
    This function count the number of times each word in
    wd_list occur in the text. To be more precise, occ_list[i]
    will give the number of times the ith word of wd_list occurs
    in the text.

    INPUT: text (string)
           wd_list (list of string)

    OUTPUT: occ_list (list of int) 
    """
    L = len(wd_list)
    occ_list = [0]*L
    
    # complete

    return occ_list


# Try this to debug
text = "the I like I like like ha ha like the pineapple the the the"
wl = ['the', 'like', 'I']
print(count_words(text, wl))


# Then try this 
common_words = ["a", "The", "the", "is", "are", "in", "In"]

text = "In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.[1] Colloquially, the term artificial intelligence is often used to describe machines (or computers) that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving. As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect.[3] A quip in Tesler's Theorem says AI is whatever hasn't been done yet. For instance, optical character recognition is frequently excluded from things considered to be AI[5] , having become a routine technology.[6] Modern machine capabilities generally classified as AI include successfully understanding human speech,[7] competing at the highest level in strategic game systems (such as chess and Go),[8] autonomously operating cars, intelligent routing in content delivery networks, and military simulations."

print(count_words(text, common_words))


"""
PROBLEM 2

Write a function that takes as input a list and returns as output the same list except that all duplicate items are removed. Here is an example:

input: ["lion", "dog", "cat", "dog"]
output: ["lion", "dog", "cat"]

Here is another example

input:  [1,5,2,3,2,1,10,15,2,100]
output: [1,5,2,3,10,15,100]

You are not allowed to use the "in" operator (asin: "if x in lst"). Instead you need to use nested loop.

"""

def remove_duplicate(lst):

    new_lst = []

    # complete

    return new_lst


lst1 = ["lion", "dog", "cat", "dog"]
print(remove_duplicate(lst1))

lst2 = [1,5,2,3,2,1,10,15,2,100]
print(remove_duplicate(lst2))

"""
PROBLEM 3

Implement your own split function. It should take a text as input, and return a list of the word contained in this text. To do this start with an empty list, and an empty string.

You will want to use string concatenation to build words by adding characters one after the other. Here is an example:

wd = "Hell"
wd = wd + "o"
print(wd)

And if you hit the " " character, this means it is the end of the word you are building, and you need to append it to the list, and reinitialise your word to be the empty string.

"""

def my_split(text):
    """
    INPUT: text (string)
    OUTPUT: lst (list of string)
    """
    lst = []
    word = ""

    # complete
    
    return lst

text = "In computer science , artificial intelligence (AI) , sometimes called machine intelligence, is intelligence demonstrated by machines , in contrast to the natural intelligence displayed by humans and animals."

lst = my_split(text)
print(lst)




