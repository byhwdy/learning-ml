
# coding: utf-8

# # <font color='brown'>楼 + 机器学习实战</font>
# 
# # 挑战：高斯分布函数实现及绘图

# ## 挑战介绍

# 朴素贝叶斯实验中提到了高斯分布，本次挑战通过 Python 实现高斯分布函数，并使用 Matplotlib 绘制不同参数下的高斯分布图像。

# ## 挑战知识点

# - 实现高斯分布
# - 绘制高斯分布图像

# ---

# ## 挑战内容

# ### 高斯分布

# 在朴素贝叶斯的实验中，我们知道可以依照特征数据类型，在计算先验概率时对朴素贝叶斯模型进行划分，并分为：多项式模型，伯努利模型和高斯模型。而在前面的实验中，我们使用了多项式模型来完成。

# 很多时候，当我们的特征是连续变量时，运用多项式模型的效果不好。所以，我们通常会采用高斯模型对连续变量进行处理，而高斯模型实际上就是假设连续变量的特征数据是服从高斯分布。其中，高斯分布函数表达式为：

# $$
# P=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu) ^{2}}{2\sigma ^{2}})
# $$

# 其中 $\mu$ 为均值，$\sigma$ 为方差。

# ---

# **<font color='red'>挑战</font>：参考高斯分布公式，使用 Python 实现高斯分布函数。**

# In[17]:


"""实现高斯分布函数
"""
import numpy as np

def Gaussian(x, u, d):
    """
    参数:
    x -- 变量
    u -- 均值
    d -- 方差

    返回:
    p -- 高斯分布值
    """
    ### 代码开始 ### (≈ 3~5 行代码)
    a = 1 / (np.sqrt(2*np.pi)*d)
    b = np.exp(-np.square(x-u)/(2*(d*d)))
    p = a*b
    
    return p
    ### 代码结束 ###


# **运行测试：**

# In[18]:


x = np.linspace(-5, 5, 100)
u=3
d=1
g = Gaussian(x, u, d)

len(g),g[10]


# **期望输出：**

# <div align="center">**`(100, 0.000139341123134969)`**</div>

# 实现高斯分布函数之后，我们可以使用 Matplotlib 绘制出不同参数下的高斯分布图像。

# ---

# **<font color='red'>挑战</font>：按规定的参数绘制高斯分布图像。**

# **<font color='blue'>规定</font>**：
# 
# - 绘制 4 组高斯分布线形图像，$\mu$ 和 $\sigma$ 分别为：`(0, 1), (-1, 2), (1, 0.5), (0.5, 5)`。
# - 4 组高斯分布图像的线形颜色分别为红色、蓝色、绿色、黄色。
# - 绘制图例，并以 u=$\mu$, d=$\sigma$ 样式呈现。

# In[19]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

### 代码开始 ### (≈ 5~10 行代码)
g_p = [(0, 1), (-1, 2), (1, 0.5), (0.5, 5)]
c = ['r', 'b', 'g', 'y']
x_temp = np.linspace(-5, 5, 100)
legend_label = []
# plt.figure(figsize=(16, 9))
for i, p_iterm in enumerate(g_p):
    p = Gaussian(x_temp, p_iterm[0], p_iterm[1])
    plt.plot(x_temp, p, c[i])
    legend_label.append('u={},d={}'.format(p_iterm[0], p_iterm[1]))
plt.legend(legend_label, loc=1)

### 代码结束 ###


# **期望输出：**

# ![download-1.png](attachment:download-1.png)

# ---

# <div style="color: #999;font-size: 12px;font-style: italic;">*本课程内容，由作者授权实验楼发布，未经允许，禁止转载、下载及非法传播。</div>
