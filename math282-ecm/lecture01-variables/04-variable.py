
"""
from the text: 
An assignment statement creates a new variable and gives it a value
"""

x = 30
y = 3.14
text = "hello, world"

print(x)
print(y)
print(text)


###################
x = 30
y = 50
z = (x+y)/2
print(z)
print(x,y,z)
print('z is equal to', z)


######################

temp_f = 70.0
temp_c = (temp_f - 32) * 5/9
print( 'temp in Fahrenheit:' , temp_f)
print( 'temp in Celcius:', temp_c)


###################

offset = 32
multiplier = 5/9

temp_f = 70.0
temp_c = (temp_f - offset) * multiplier

print( 'temp in Fahrenheit:' , temp_f)
print( 'temp in Celcius:', temp_c)

####################

# overwriting a variable
x = 5
y = x*2

x = 7
z = x*2

print(x,y,z)

###################
# The equal sign means "assign":

x = 10
x = 2*x
print(x)

###############
"""
Running sum. What is the value of s below?
"""

s = 0
s = s+2
s = s+10
s = s+10 
print(s)

##############
# LEt's try all this with python tutor
"""
http://pythontutor.com/
"""

a = 5
b = 10
c = 1

a = a+c
b = b+1
c = a+1 



###################
"""
Exercise 1: find the output without running the code. Then use http://pythontutor.com/ to check your reasonning is correct

x = 3
y = x*2
z = y*10 
print(z)
x = x+1
y = y+1
z = x+y+z 
print(z)


Exercise 2: Write a code that compute the final grade in a class where the grading scheme is:
hwk: 10%
mid1: 25%
mid2: 25%
final: 40%

Assume that you got the following grades: 
hwk: 95/100
mid1: 84/100
mid2: 92/100
final: 88/100

You have 8 variables (make sure to choose good names for these 8 variables!)
"""






