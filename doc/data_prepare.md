A machine learning model can't directly see, hear, or sense input examples. Instead, you must create a representation of the data to provide the model with a useful vantage point into the data's key qualities. That is, in order to train a model, you must choose the set of features that best represent the data.

In traditional programming, the focus is on code. In machine learning projects, the focus shifts to representation. That is, one way developers hone a model is by adding and improving its features.

## 良好特征的特点:  
?

## 一、 Feature engineering
Feature engineering means transforming raw data into a feature vector.

Many machine learning models must represent the features as real-numbered vectors since the feature values must be multiplied by the model weights.

### 1. converting
#### 1) numeric values

#### 2) categorical values
类别特征的值把样本分成几组
##### one-hot encoding /  multi-hot encoding
问题：  
将类别值映射成整数 vocabulary, OOV (out-of-vocabulary) bucket
1. 不合理 
2. 无法表示多个值  
one-hot encoding /  multi-hot encoding

#### 3) sparse representation  
?

### 2. dropping

### 3. creating
#### synthetic feature
1. feature crosses  
对非线性规律进行编码  
实践中，最常用的是**Crossing One-Hot Vectors**，可以将独热特征矢量的特征组合视为逻辑连接, 例如：country:usa AND language:spanish  
线性学习器可以很好地扩展到大量数据。对大规模数据集使用特征组合是学习高度复杂模型的一种有效策略。神经网络可提供另一种策略


## 二、 Cleaning Data
As an ML engineer, you'll spend enormous amounts of your time tossing out bad examples and cleaning up the salvageable ones.

#### Scaling
作用：  
- 帮助梯度下降法更快速地收敛。
- 帮助避免“NaN 陷阱”。在这种陷阱中，模型中的一个数值变成 NaN（例如，当某个值在训练期间超出浮点精确率限制时），并且模型中的所有其他数值最终也会因数学运算而变成 NaN。
- 帮助模型为每个特征确定合适的权重。如果没有进行特征缩放，则模型会对范围较大的特征投入过多精力
不需要对每个浮点特征进行完全相同的缩放, 但是差别也不能太大

#### outliers
长尾巴分布  
方法：  
- 取对数
- 规定最大值

#### Binning
在数据集中，latitude是一个浮点值。不过，在我们的模型中将latitude表示为浮点特征没有意义。这是因为纬度和房屋价值之间不存在线性关系。例如，纬度 35 处的房屋并不比纬度 34 处的房屋贵 35/34（或更便宜）。但是，纬度或许能很好地预测房屋价值。
分箱之后，我们的模型现在可以为每个纬度学习完全不同的权重。

#### bad data
在现实生活中，数据集中的很多样本是不可靠的，原因有以下一种或多种：

- 缺失值。 例如，有人忘记为某个房屋的年龄输入值。
- 重复样本。 例如，服务器错误地将同一条记录上传了两次。
- 不良标签。 例如，有人错误地将一颗橡树的图片标记为枫树。
- 不良特征值。 例如，有人输入了多余的位数，或者温度计被遗落在太阳底下

要检测缺失值或重复样本，您可以编写一个简单的程序。检测不良特征值或标签可能会比较棘手。




