import time
def countdown(n):
	while n > 0:
		print('T-minus', n)
		n -= 1
		time.sleep(2)

from threading import Thread 
t = Thread(target=countdown, args=(10,))
t.start()

n = 15
while n > 0: 
	if t.is_alive():
		print('still running')
	else:
		print('completed')
	n -= 1
	time.sleep(2)
