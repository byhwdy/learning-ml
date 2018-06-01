from random import random

def estimate_pi(times):
	hits = 0
	for _ in range(times):
		x = random()*2 - 1
		y = random()*2 - 1
		if x ** 2 + y ** 2 <= 1:
			hits += 1
	return 4.0 * hits / times

if __name__ == '__main__':
	print(estimate_pi(10000))
	print(estimate_pi(1000000))
	print(estimate_pi(100000000))
	print(estimate_pi(1000000000))