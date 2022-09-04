# A simple example:
n = 4

while n > 0:
    n = n-1
    print(n)

print('over!')

# Slightly nore complicated
n = 50
while n > 0:
	if n%10 == 0:
		n = n - 5
    else:
    	n = n-1
    print(n)


# Find the smallest fibonnacy numbers which is greater than 100
x_old = 0
x_cur = 1

while x_cur < 100:
	x_new = x_old + x_cur
	x_old = x_cur
	x_cur = x_new
	print(x_cur)


# Exercise: find the smallest n such that n^2 > 1000
