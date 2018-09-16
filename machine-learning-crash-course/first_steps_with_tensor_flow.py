from __future__ import print_function
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn import metrics 
import tensorflow as tf 
from tensorflow.python.data import Dataset

# 设置
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format 

# 数据集
california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")

# 对数据进行随机化处理，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）。
# 此外，我们会将 median_house_value 调整为以千为单位，这样，模型就能够以常用范围内的学习速率较为轻松地学习这些数据
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
# print(california_housing_dataframe.head())

## Step 1: Define Features and Configure Feature Columns
# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[['total_rooms']]
# print(my_feature)

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column('total_rooms')]

## Step 2: Define the Target
# Define the label
targets = california_housing_dataframe['median_house_value']

## Step 3: Configure the LinearRegressor
# Use gradient descent as the optimizer for training the model.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

## Step 4: Define the Input Function
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

## Step 5: Train the Model
_ = linear_regressor.train(
    input_fn=lambda:my_input_fn(my_feature, targets),
    steps=100
)

## Step 6: Evaluate the Model
# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

predictions = linear_regressor.predict(input_fn=prediction_input_fn)
# print(list(predictions))

predictions = np.array([item['predictions'][0] for item in predictions])
# print(predictions)

# Print Mean Squared Error and Root Mean Squared Error.
# print(type(predictions), type(targets))
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("MSE (on train set): %0.3f" % mean_squared_error)
print("RMSE (on train set): %0.3f" % root_mean_squared_error)

# Let's compare the RMSE to the difference of the min and max of our targets:
min_house_value = california_housing_dataframe['median_house_value'].min()
max_house_value = california_housing_dataframe['median_house_value'].max()
min_max_difference = max_house_value - min_house_value
print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("RMSE: %0.3f" % root_mean_squared_error)

california_data = pd.DataFrame()
california_data['predictions'] = pd.Series(predictions)
california_data['targets'] = pd.Series(targets)
print(california_data.describe())

sample = california_housing_dataframe.sample(n=300)
x_0 = sample['total_rooms'].min()
x_1 = sample['total_rooms'].max()
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias
plt.plot([x_0, x_1], [y_0, y_1], c='r')
plt.ylabel('median_house_value')
plt.xlabel('total_rooms')
plt.scatter(sample['total_rooms'], sample['median_house_value'])
plt.show()

