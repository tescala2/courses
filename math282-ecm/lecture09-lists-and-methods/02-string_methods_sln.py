#########################################################
# Some other methods
#########################################################
text = "I like pineaple. I also really like coconuts!"
new_text = text.upper()
print(new_text)


text = "I like pineaple. I also really like coconuts!" 
new_text = text.replace('like', 'love')
print(new_text)

text = "I like pineaple. I also really like coconuts!" 
num = text.count('like')
print(num)

text = "I like pineaple. I also really like coconuts!" 
num = text.count('al')
print(num)

#########################################################
# Going back to the split() method
#########################################################

text = "I like pineaple. I also really like coconuts!"

mylist = text.split(" ")
print(mylist)

mylist = text.split(".")
print(mylist)

sentence = "aabb-zzzzz-Oh!"
mylist = text.split("-")
print(mylist)

#################################
# The join() method
#################################

mychar = "-*-"

fruits = ["banana", "apple", "orange", "coconuts"]

text = mychar.join(fruits)

print(text)

# Remark: the following would have feel more natural:
# but split and join are methods for strings

text = fruits.join(char)


# other example:

fruits = ["banana", "apple", "orange", "coconuts"]

text = "_".join(fruits)
print(text)

text = "".join(fruits)
print(text)

text = " ".join(fruits)
print(text)

text = "\t".join(fruits)
print(text)

# exercise 1: How many words and sentences in this text?

text = "Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. Reinforcement learning differs from supervised learning in not needing labelled input/output pairs be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge). The environment is typically stated in the form of a Markov decision process (MDP), because many reinforcement learning algorithms for this context utilize dynamic programming techniques. The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible."

sentences = text.split(".")
words = text.split(" ")

print('num char = ', len(text))
print('num words = ', len(words))
print('num sentences = ', len(sentences))

# exercise 2: Create a new string, called truncated_text, that only contains the first 10 words. You will need to use both split and join.

mylist = text.split(" ")
truncated_list = mylist[0:10]
truncated_text = " ".join(truncated_list)

# exercise 3: Create a new string, called text_with_big_space, in which all spaces are replaced with a tab. DO NOT USE THE REPLACE METHOD.

mylist = text.split(" ")
text_with_big_space = "\t".join(mylist)

# exercise 4: Complete the function below that count the number of occurences of  the given word into the given text. DO NOT USE THE COUNT METHOD. 

def count_word(text, word):
    """
    This function count the number of times word is appearing in text
    INPUT: text (string)
           word (string)
    OUTPUT: num_occ (int) 
    """

    mylist = text.split(" ")
    L = len(mylist)

    num_occ = 0
    for i in range(0,L):
        if mylist[i] == word:
            num_occ = num_occ+1

    return num_occ

print( count_word(text,"learning") )


# Remark: the count() do the same:
print(text.count(" learning "))

