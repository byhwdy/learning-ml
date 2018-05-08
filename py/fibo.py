# 斐波那契数列模块

def fib(n):   # write
	a, b = 0, 1
	while b < n:
		print(b, end=' ')
		a, b = b, a+b
	print()

def fib2(n):  # return
	result = []
	a, b = 0, 1
	while b < n:
		result.append(b)
		a, b = b, a+b
	return result

_myname = '私有变量'

if __name__ == '__main__':
	import sys
	fib(int(sys.argv[1]))