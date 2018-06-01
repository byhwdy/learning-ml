#### 迭代器
# class my_iterater:
# 	def __init__(self):
# 		self.lim = 10

# 	def __iter__(self):
# 		return self

# 	def __next__(self):
# 		current_value = self.lim
# 		self.lim -= 1
# 		if self.lim > 0:
# 			return 10 - current_value
# 		else:
# 			raise StopIteration

# 生成器，yield
# def generate_square(n):
# 	i = 0
# 	while i < n:
# 		yield i * i 
# 		i += 1

# def generate_square(n):
# 	i = 0 
# 	result = []
# 	while i < n:
# 		result.append(i * i)
# 		i += 1
# 	return result

if __name__ == '__main__':
	# yield			
	result = generate_square(10)
	print(list(result))