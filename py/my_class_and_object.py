class Vehicle:
	def __init__(self, number_of_wheels, type_of_tank, seating_capacity, maximum_velocity):
		self.number_of_wheels = number_of_wheels
		self.type_of_tank = type_of_tank
		self.seating_capacity = seating_capacity
		self.maximum_velocity = maximum_velocity

	@property
	def number_of_wheels(self):
		return self.number_of_wheels

	@number_of_wheels.setter
	def number_of_wheels(self, number):
		self.number_of_wheels = number



# f = tesla_model_s.say
# f()
# f = Vehicle.say
# f()
tesla_model_s = Vehicle(4, 'electric', 5, 250)
# print(tesla_model_s)