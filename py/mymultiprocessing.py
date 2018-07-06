import os

res = os.fork()
if res == 0:
	print('这是子进程%s', os.getpid())
else:
	print('这是父进程%s', os.getppid())