#### 迭代器
class my_iterater:
	def __init__(self):
		self.lim = 10

	def __iter__(self):
		return self

	def __next__(self):
		current_value = self.lim
		self.lim -= 1
		if self.lim > 0:
			return 10 - current_value
		else:
			raise StopIteration

			