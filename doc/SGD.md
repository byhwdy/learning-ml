### GD(gradient descent)
优化损失函数的方法  

**术语**：  
**step**：  
**batch**(批量)：指的是用于在单次迭代中计算梯度的样本总数
**Convergence**（收敛）：

**特征**：
1. 一种迭代方法
2. 梯度
3. 超参数：学习速率learning rate  
    梯度下降法算法用梯度乘以一个称为学习速率（有时也称为步长）的标量，以确定下一个点的位置。
    - 学习速率过小，就会花费太长的学习时间
    - 学习速率过大，下一个点将永远在 U 形曲线的底部随意弹跳
    - 对每个特定问题，都有一个Goldilocks learning rate。“金发姑娘”值与损失函数的平坦程度相关。如果您知道损失函数的梯度较小，则可以放心地试着采用更大的学习速率，以补偿较小的梯度并获得更大的步长。

    **实践：优化学习速率**  
    在实践中，成功的模型训练并不意味着要找到“完美”（或接近完美）的学习速率。我们的目标是找到一个足够高的学习速率，该速率要能够使梯度下降过程高效收敛，但又不会高到使该过程永远无法收敛。

### SGD(Stochastic Gradient Descent)
批量过大，则单次迭代就可能要花费很长时间进行计算  
超大批量所具备的预测价值往往并不比大型批量高  
What if we could get the right gradient on average for much less computation? By choosing examples at random from our data set, we could **estimate (albeit, noisily) a big average from a much smaller one**.

**Stochastic gradient descent (SGD)**: it uses only a single example (a batch size of 1) per iteration.  
- Given enough iterations, SGD works but is very **noisy**.

**Mini-batch stochastic gradient descent (mini-batch SGD)**: a compromise between full-batch iteration and SGD.  
- A mini-batch is typically between 10 and 1,000 examples, chosen at random.  
- Mini-batch SGD **reduces the amount of noise** in SGD but is still more efficient than full-batch.

**实践**： 了解模型收敛所需的步数，并了解模型收敛的顺滑平稳程度