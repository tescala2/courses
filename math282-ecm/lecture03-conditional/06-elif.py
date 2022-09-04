def compare(x,y):
	if x < y:
	    print('x is less than y')
	elif x > y:
	    print('x is greater than y')
	else:
	    print('x and y are equal')

x=10
y=17
compare(x,y)

#############################################
# this doesn't work!

def greet(time):
	"""
	print a time dependent greeting
	"""
	if time < 11:
		print("Good morning")
	if time <13: 
		print("Did you have lunch yet?")
	if time <18: 
		print("Good day")
	if time <22: 
		print("Good evening")
	if time <24: 
	 	print("Good night")

time = 12
greet(time)

#############################
# This is what we need:

def greet(time):
	"""
	print a time dependent greeting
	"""
	if time < 11:
		print("Good morning")
	if time >= 11 and time <13: 
		print("Did you have lunch yet?")
	if time >= 13 and time <18: 
		print("Good day")
	if time >= 18 and time <22: 
		print("Good evening")
	if time >= 22 and time <24: 
	 	print("Good night")

time = 12
greet(time)

##########################
# or better:

def greet(time):
	"""
	print a time dependent greeting
	"""
	if time < 11:
		print("Good morning")
	elif time <13: 
		print("Did you have lunch yet?")
	elif time <18: 
		print("Good day")
	elif time <22: 
		print("Good evening")
	else: 
	 	print("Good night")

time = 12
greet(time)


#############################
############################
def old_enough(birth_date, current_year):
    """
    return True if the customer is 21 year old or more
    """
    age = current_year - birth_date
    
    if age>21:
        print('customer is old enough')
    elif age<21:
    	print('customer is not old enough')
    else:
    	print('need the exact bith date')

old_enough(1998,2020)





