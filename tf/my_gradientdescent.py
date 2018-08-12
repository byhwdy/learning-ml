"""
用tf练习gd
求 y= 3x^2 + 5 的最小值
"""
import tensorflow as tf 

learning_rate = 0.2
n_epoches = 1000

x = tf.Variable(tf.random_uniform([1], seed=32)*100)
y = 3 * tf.square(x) + 50
grad = 6 * x
gd_op = tf.assign(x, x - learning_rate * grad)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epoches):
		if epoch % 100 == 0:
			print('epoch {} is running'.format(epoch))
		sess.run(gd_op)
	result = y.eval()

print(result)