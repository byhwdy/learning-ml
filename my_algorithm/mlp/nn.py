import numpy as np
import loaddata

# class FullConnectedLayer():
# 	"""全连接层实现类"""

# 	def __init__(self, input_size, output_size, activator):
# 		"""
# 		input_size: 本层输入向量的维度
# 		output_size: 本层输出向量的维度
# 		activator： 激活函数
# 		"""
# 		self.input_size = input_size
# 		self.output_size = output_size
# 		self.activator = activator
# 		# 权重矩阵
# 		self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
# 		# 偏置项b
# 		self.b = np.zeros((output_size, 1))
# 		# 输出向量
# 		self.output = np.zeros((output_size, 1))

# 	def forward(self, input_array):
# 		"""
# 		前向传播
# 		input_array: 输入向量，维度必须等于input_size
# 		"""
# 		self.input = input_array
# 		self.output = self.activator.forward(
# 			np.dot(self.W, input_array) + self.b)

# 	def backward(self, delta_array):
# 		"""
# 		反向计算W和b的梯度
# 		delta_array: 从上一层传递过来的误差项
# 		"""
# 		self.delta = self.activator.backward(self.input) * np.dot(
# 			self.W.T, delta_array)
# 		self.W_grad = np.dot(delta_array, self.input.T)
# 		self.b_grad = delta_array

# 	def update(self, learning_rate):
# 		"""
# 		使用梯度下降算法更新权重
# 		"""
# 		self.W += learning_rate * self.W_grad
# 		self.b += learning_rate * self.b_grad


# class SigmoidActivator():
# 	"""sigmoid激活函数类"""
# 	def forward(self, weighted_input):
# 		return 1.0 / (1.0 + np.exp(-weighted_input))

# 	def backward(self, output):
# 		return output * (1 - output)

# class Netword():
# 	"""神经网络类"""
# 	def __init__(self, layers):
# 		self.layers = []
# 		for i in range(len(layers) - 1):
# 			self.layers.append(
# 				FullConnectedLayer(
# 					layers[i], layers[i+1],
# 					SigmoidActivator()
# 				)
# 			)

# 	def predict(self, sample):
# 		output = sample
# 		for layer in self.layers:
# 			layer.forward(output)
# 			output = layer.output
# 		return output

# 	def train(self, labels, data_set, rate, epoch):
# 		for i in range(epoch):
# 			for d in range(len(data_set)):
# 				self.train_one_sample(labels[d],
# 					data_set[d], rate)

# 	def train_one_sample(self, label, sample, rate):
# 		self.predict(sample)
# 		self.calc_gradient(label)
# 		self.update_weight(rate)

# 	def calc_gradient(self, label):
#         delta = self.layers[-1].activator.backward(
#             self.layers[-1].output
#         ) * (label - self.layers[-1].output)
#         for layer in self.layers[::-1]:
#             layer.backward(delta)
#             delta = layer.delta
#         return delta
        
#     def update_weight(self, rate):
#         for layer in self.layers:
#             layer.update(rate)

def get_result(vec):
	"""从标签向量获取值"""
	max_value_index = 0
	max_value = 0
	for i, v in enumerate(vec):
		if v > max_value:
			max_value = v
			max_value_index = i 
	return max_value_index

def evaluate(network, test_data_set, test_labels):
	"""模型评价方法：预测准确率"""
	
	error = 0
	total = len(test_data_set)

	for i in range(total):
		label = get_result(test_labels[i])
		predict = get_result(network.predict(test_data_set[i]))
		if label != predict:
			error += 1

	return float(error) / float(total)



if __name__ == '__main__':
	print(get_result([1,2,3,4,5,4,3]))