import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 

# 数据
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig = plt.figure()
# ax = fig.add_subplot(111)
ax = fig.add_axes([0.1, 0.2, 0.8, 0.6])

# fig, ax = plt.subplots()

ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
	   title='About as simple as it gets, folks')
ax.grid()

fig.savefig('test.png')
plt.show()