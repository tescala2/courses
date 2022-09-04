def is_multiple_of_five(n):
	b = n%5 == 0
	if b:
		print(n, 'is a multiple of 5')
	else:
		print(n, 'is not a multiple of 5')
	
is_multiple_of_five(102)