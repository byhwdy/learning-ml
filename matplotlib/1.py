from matplotlib import pyplot as plt
import numpy as np 

x = np.linspace(0, 10, 20)

fig, axes = plt.subplots()
# x,y轴，图例
# axes.set_xlabel('X')
# axes.set_ylabel('Y')
# axes.set_title('TITLE')
axes.plot(x, x**2)
axes.plot(x, x**3)
axes.legend(['y = x**2', 'y = x**3'], loc=2)
plt.show()