import threading
import time

def profile(func):
	def wrapper(*args, **kwargs):
		import time
		start = time.time()
		func(*args, **kwargs)
		end = time.time()
		print('cost: {}'.format(end-start))
	return wrapper

def fib(n):
	if n <= 2:
		return 1
	return fib(n-1) + fib(n-2)

if __name__ == '__main__':
	