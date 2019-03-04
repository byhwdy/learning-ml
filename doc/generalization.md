## generalization

Generalization refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.

## 过拟合
An overfit model gets a low loss during training but does a poor job predicting new data.

Overfitting is caused by making a model more complex than necessary. The fundamental tension of machine learning is between fitting our data well, but also fitting the data as simply as possible

## 问题
how can you trust the model will also make good predictions on never-before-seen examples?

## 理论
#### 奥卡姆剃刀原理在机器学习中的应用
The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.

#### statistical learning theory and computational learning theory
In modern times, we've formalized Ockham's razor into the fields of **statistical learning theory** and **computational learning theory**. These fields have developed generalization bounds--a statistical description of a model's ability to generalize to new data based on factors such as:

- the complexity of the model
- the model's performance on training data

## 实证方法（划分出测试集）
虽然理论分析在理想化假设下可提供正式保证，但在实践中却很难应用。机器学习速成课程则侧重于实证评估，以评判模型泛化到新数据的能力。

A machine learning model aims to make good predictions on new, previously unseen data. But if you are building a model from your data set, how would you get the previously unseen data? Well, one way is to divide your data set into two subsets:
- training set—a subset to train a model.
- test set—a subset to test the model.

Good performance on the test set is a useful indicator of good performance on the new data in general, assuming that:
- The test set is large enough.
- You don't cheat by using the same test set over and over.

#### 方法：Splitting Data
Make sure that your test set meets the following two conditions:
- Is large enough to yield statistically meaningful results.
- Is representative of the data set as a whole. In other words, don't pick a test set with different characteristics than the training set.

Never train on test data


## 验证集
- We looked at a process of using a test set and a training set to drive iterations of model development. On each iteration, we'd train on the training data and evaluate on the test data, using the evaluation results on test data to guide choices of and changes to various model hyperparameters like learning rate and features. Is there anything wrong with this approach?
- Doing many rounds of this procedure might cause us to implicitly fit to the peculiarities of our specific test set.

"Tweak model" means adjusting anything about the model you can dream up—from changing the learning rate, to adding or removing features, to designing a completely new model from scratch. At the end of this workflow, you pick the model that does best on the test set.

