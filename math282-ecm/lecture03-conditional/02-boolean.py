# We have a new type!

x = True
y = False

print(x)
print(y)
print( type(x) )


################
# Do operation on it

x = False
n = 3

m = - n
print(m)

y = not x
print(y)

################
# and / or


m = 2
n = 3
t = m + n
print(t)

x = True
y = False
z = x or y
print(z)

w = y or y
print(y)

w = y or (not y)
print(w)

###########################
# precedence not > and > or

a = True
b = False
c = False

d = a and b or a and not c
print(d)

d = (a and b) or (a and (not c) )
print(d)

##################
# exercise: figure out the output without running the code
# then run the code to check

x = True
y = False

z = not x and x or not x and y
print(z)

v = not x and x or not (x and y)
print(v)

w = not ( x and x or x and y)
print(w)






