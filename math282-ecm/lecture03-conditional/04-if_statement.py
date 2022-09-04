"""
 if must be followed by a boolean variable.
 If the boolean is equal to True then all the indented statement will be executed
 If the boolean statement is equal to False, then we skip the block
"""
def greet(name,b):
    """
    This function greet in 2 different ways
    depending on the boolean
    input: name (string)
           b (boolean)
    """
    print("Hi", name)

    if b:
        print('I am so happy to see you!')
        print('You are favorite person in the world!') 
        
    print('How are you doing?')



greet('Elizabeth', True)


##########################
""" 
Increase readibility by giving
more meaningful name to the boolean argument
"""

def greet(name,excited):
    """
    This function greet in 2 different ways
    depending of the boolean
    input: name (string)
           excited (boolean)
    """
    print("Hi", name)
    if excited:
        print('I am so happy to see you!')
        print('You are favorite person in the world!')
    print('How are you doing?')
    
   
greet('Elizabeth', True)


#############
def old_enough(birth_date, current_year):
    """
    return True if the customer is 21 year old or more
    """

    output = False
    age = current_year - birth_date
    b = age >= 21
    if b:
        output = True

    return output

OK = old_enough(1998,2020)
print(OK)

#############
def verify_age(birth_date, current_year):
    """
    return True if the customer is between 18 and 90 years old
    """

    output = False
    age = current_year - birth_date
    b = age >= 18 and age <= 90
    if b:
        output = True

    return output

OK = verify_age(2005,2020)
print(OK)

###############
# this is more readable to directly put the comparison in the if statement

def verify_age(birth_date, current_year):
    """
    return True if the customer is between 18 and 90 years old
    """

    output = False
    age = current_year - birth_date
    if age >= 18 and age <= 90:
        output = True

    return output

OK = verify_age(2005,2020)
print(OK)


##############
def add_one_and_double(x,max):
    """
    This function add 1 to x and then multiply the result by 2 as long as x is smaller than max 
    """
    if x<max:
        x = x+1
        x = x*2
    return x

M = 100
x = 3

x = add_one_and_double(x,M)
print(x)

x = add_one_and_double(x,M)
print(x)

x = add_one_and_double(x,M)
print(x)

x = add_one_and_double(x,M)
print(x)

x = add_one_and_double(x,M)
print(x)

x = add_one_and_double(x,M)
print(x)



###############
# Here is how you access the last character of a string:

sentence = "My name is Thomas"
x = sentence[-1]
print(x)
print(type(x))


###############
"""
Exercice: Create a function called add_exclamation that adds an exclamation point at the end of a string except if the last letter of the string is already an exclamation point.
"""




