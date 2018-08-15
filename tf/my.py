import numpy as np 
import tensorflow as tf 


def my_input_fn():
	ds = tf.data.Dataset.from_tensor_slices(np.arange(1000).reshape(100, 10))
	iterator = ds.make_one_shot_iterator()
	next_element = iterator.get_next()
	return next_element

next_element = my_input_fn()

n = 0

1 + 1


for i in range(5):
	n += next_element

n + 2

# graphboard
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

if __name__ == '__main__':
	with tf.Session() as sess:
		# for i in range(10):
		# 	print(sess.run(next_element))
		print(sess.run(n))