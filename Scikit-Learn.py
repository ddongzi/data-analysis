# -*- coding: utf-8 -*-
""" 《python数据科学》Sckit-Learn部分
已过时，参考Scikit-Lean
Created on Thu Mar 20 23:29:12 2025

@author: 63517
"""

# ## 5.2　Scikit-Learn简介
# 是一个机器学习算法库，特点 管道式命令api
#

# ### 5.2.1 数据表示
# - 数据表示为DataFrame

# 1. 数据表：
#    每行表示一个样本，列表示特征

# In[1]:


from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from ipywidgets import interact, fixed
from sklearn.svm import SVC  # "Support vector classifier"
from sklearn.datasets import make_blobs
from scipy import stats
from numpy import nan
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import learning_curve
import seaborn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import Isomap
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA  # 1.选择模型类
from sklearn.naive_bayes import GaussianNB  # 1.选择模型类
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import precision_score, recall_score, f1_score
import jieba
from sklearn.model_selection import train_test_split  # 分割数据集
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups  # 2000篇新闻
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs  # 创建几类随机数据
from sklearn.datasets import fetch_20newsgroups
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture  # 1.选择模型类
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # 自动分割数据集
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
iris = sns.load_dataset('iris')
iris.describe()


# 2. 特征矩阵：每行是对象，每列都是量化值。[n_samples, n_features]
# 3. 目标（标签）数组：[n_samples, n_targets] 一般是二维数组。

# In[73]:


sns.pairplot(iris)  # 快速查看各特征关系


# In[2]:


X_iris = iris.drop('species', axis=1)
X_iris.shape
Y_iris = iris['species']
Y_iris.shape


# In[ ]:


# ### 5.2.2　Scikit-Learn的评估器API

# **1. API基础知识**
#
# Scikit-Learn 评估器 API 的常用步骤如下所示（后面介绍的示例都是按照这些步骤进行的
#
# (1) 通过从 Scikit-Learn 中导入适当的评估器类，*选择模型类*。
#
# (2) 用合适的数值对模型类进行实例化，配置模型超参数（hyperparameter）。
#
# (3) 整理数据，通过前面介绍的方法获取特征矩阵和目标数组。
#
# (4) 调用模型实例的 fit() 方法对数据进行拟合。
#
# (5) 对新数据应用模型：
#
# - 在有监督学习模型中，通常使用 predict() 方法预测新数据的标签；
# - 在无监督学习模型中，通常使用 transform()或 predict()方法转换或推断数据的性质
#
# 下面按照步骤来演示几个使用了有监督学习方法和无监督学习方法的示例。

# In[ ]:


# **2. 有监督学习示例：简单线性回归**

# (0) 导入数据

# In[3]:


rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.rand(50)
plt.scatter(x, y)


# (1) 选择模型类

# In[5]:


# (2) 配置模型实例对象的超参数

# In[6]:


model = LinearRegression(fit_intercept=True)  # 带截距


# (3) 整理数据为特征矩阵和目标数组

# In[8]:


X = x[:, np.newaxis]  # 转换成列
X.shape


# (4) 模型拟合数据
# - fit() 命令会在模型内部进行大量运算，运算结果将存储在模型属性中
# - 获得的模型参数都是后缀_的

# In[9]:


model.fit(X, y)


# In[10]:


model.coef_  # 斜率


# In[11]:


model.intercept_  # 截距


# (5) 预测新数据

# In[12]:


xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit)  # 连接成线


# **3. 有监督学习示例：鸢尾花数据分类**，高斯朴素贝叶斯（Gaussian naive Bayes）

# In[14]:


iris.loc[:, :'species']


# 假设样本是由高斯分布产生的，用朴素贝叶斯算法分类。
#
# 通常都会先用这个算法，因为速度很快，不需要超参数配置。然后优化

#

# In[20]:


# (0)导入数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_iris, Y_iris, random_state=1)
print(Xtrain, Ytrain)


# In[22]:


model = GaussianNB()                       # 2.初始化模型
model.fit(Xtrain, Ytrain)                  # 3.用模型拟合数据
y_model = model.predict(Xtest)             # 4.对新数据进行预测


# In[24]:


accuracy_score(y_model, Ytest)  # 检查模型准确率


# **4. 无监督学习示例：鸢尾花数据降维**

# In[25]:


model = PCA(n_components=2)      # 2.设置超参数，初始化模型
model.fit(X_iris)                # 3.拟合数据，注意这里不用y变量
X_2D = model.transform(X_iris)   # 4. 将数据转换为二维


# In[26]:


iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]


# In[27]:


iris


# In[29]:


sns.lmplot(x="PCA1", y="PCA2", hue='species', data=iris, fit_reg=False)


# 可以看到，通过PCA1/2 对setosa很好的分类， vir和ver类有点不太好

# **5. 无监督学习示例：鸢尾花数据聚类**，高斯混合模型

# In[31]:


model = GaussianMixture(
    n_components=3, covariance_type='full')  # 2.设置超参数，初始化模型
model.fit(X_iris)  # 3.拟合数据，注意不需要y变量
y_gmm = model.predict(X_iris)  # 4. 确定簇标签
y_gmm


# In[32]:


iris['cluster'] = y_gmm
iris


# In[33]:


sns.lmplot(x="PCA1", y="PCA2", hue='species',
           col='cluster', data=iris, fit_reg=False)


# 图上聚类和结果和PCA看来差不多，在vir和ver之间还是有点模糊

# ### 5.2.3　应用：手写数字探索

# 1. 加载并可视化手写数字

# In[34]:


digits = load_digits()
digits.images.shape


# In[39]:


fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.subplots_adjust(hspace=0, wspace=0)
for i, ax in enumerate(axes.flat):  # axes二维扁平化，方便遍历
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    # 相对坐标x,y ,显示文字， transform表示以子ax表示而不是整图
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')


# 构造特征矩阵和目标数组

# In[43]:


X = digits.data  # 已经构建好
X.shape


# In[60]:


y = digits.target
y.shape


# **2. 无监督学习：降维**
#
# 我想特征可视化先看看， 但是64维 ，维度太高了，需要降维到低维度，通过 流形学习的losmap进行降维：

# In[47]:


iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape


# In[56]:


plt.scatter(x=data_projected[:, 0], y=data_projected[:, 1], c=digits.target,
            edgecolors='none', alpha=0.5, cmap=plt.colormaps['magma'].resampled(10))
plt.colorbar(label='digit label', ticks=range(10))  # colorbar显示信息
plt.clim(-0.5, 9.5)


# 虽然看起来很混淆，毕竟10个类型，但可以看到明显有些颜色不会重叠，比如黄色9 和 6， 即区分比较好，常识来讲也是更容易区分。

# 用有监督的学习来看看，毕竟无监督看起来还可以了

# **3. 数字分类：有监督的学习**

# In[63]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


# In[65]:


model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)


# In[67]:


accuracy_score(ytest, y_model)


# 准确率指标不够，需要其他指标：混淆矩阵

# In[68]:


mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')


# 可以看到，主要是将2误判成8。 现在我们从图像上标识看下误判的。

# In[90]:


print(Xtest.shape)
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
test_images = Xtest.reshape(-1, 8, 8)
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == y_model[i]) else 'red')  # 正确的绿色标签，


# 仅查看分类错误的图像

# In[91]:


print(Xtest.shape, ytest.shape)
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
mask = ytest != y_model  # 得到错误的布尔索引
print(test_images[mask].shape)
Xtest_error = Xtest[mask]
error_ymodel = y_model[mask]
error_ytest = ytest[mask]
error_images = Xtest_error.reshape(-1, 8, 8)

plt.setp(axes, xticks=[], yticks=[])  # 一次性去掉全部坐标轴
for i, ax in enumerate(axes.flat[: len(error_images)]):
    ax.imshow(error_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(error_ymodel[i]),
            transform=ax.transAxes, c='red')  # 预测值
    ax.text(0.05, 0.5, str(error_ytest[i]),
            transform=ax.transAxes, c='green')  # 原值


# ## 5.3　超参数与模型验证
# **交叉检验方法**调整参数至关重要，这样做可以避免较复杂 / 灵活模型引起的过拟合问题。
#
# 模型验证：预测值和实际值差异。 交叉验证！
#
# 超参数优化：验证曲线。

# 1. 错误的模型验证方法

# In[92]:


iris = load_iris()
X = iris.data
y = iris.target


# In[93]:


model = KNeighborsClassifier(n_neighbors=1)


# In[94]:


model.fit(X, y)
y_model = model.predict(X)


# In[96]:


accuracy_score(y, y_model)  # 显然错误准确率，


# 2. 模型验证正确方法：留出集

# In[98]:


# 每个数据集分一半数据
X1, X2, y1, y2 = train_test_split(X, y, random_state=0,
                                  train_size=0.5)
# 用模型拟合训练数据
model.fit(X1, y1)

# 在测试集中评估模型准确率
y2_model = model.predict(X2)
accuracy_score(y2, y2_model)


# 3. 交叉检验
#
# 就是就行几轮实验，使得充分作为 训练和测试。  准确率取平均

# In[102]:


y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)


# 自动进行几轮，无须手动代码

# In[101]:


cross_val_score(model, X, y, cv=5)


# ### 5.3.2　选择最优模型：验证曲线
#
# 假如模型效果不好，应该如何改善？
#
# **问题的答案往往与直觉相悖**
#
# 换一种更复杂的模型有时可能产生更差的结果，增加更多 的训练样本也未必能改善性能！改善模型能力的高低，是区分机器学习实践者成功与否 的标志。

# **1. 偏差与方差的均衡**

# 显然，左侧图欠拟合（高偏差）（训练集和测试集都不好），右边过拟合（高方差）（训练集远远大于测试集表现）

# ![图片.png](attachment:2df6c839-3c93-42a5-853a-26b97fe763da.png)

# 通过不管调整模型复杂度，来达到折中。  不同模型调整方法不同。

# ![图片.png](attachment:5e58d71e-e755-40f1-8592-41f4e796dcb0.png)

# **2. Scikit-Learn验证曲线**
# 如何计算这样的验证曲线图，上图。

# 例子：n次多项式回归

# In[3]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


# In[4]:


def make_data(N, err=1.0, rseed=1):
    # 随机轴样数据
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)
print(X.shape, y.shape)


# In[5]:


seaborn.set()  # 设置图形样式

X_test = np.linspace(-0.1, 1.1, 500)[:, None]
plt.scatter(X.ravel(), y, color='black')  # ravel函数扁平化，返回数据视图，
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(
        X, y).predict(X_test)  # degree控制次数
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')


#

# **问题 ：哪个回归曲线更好？ 即做验证曲线，验证得分**

# ```
# validation_curve(
#     estimator,  # 需要评估的模型（如线性回归、SVM等）
#     X, y,       # 训练数据和目标值
#     param_name, # 要调整的超参数名称（字符串）
#     param_range,# 超参数的取值范围（数组/列表）
#     scoring=None, # 评分指标（默认为模型的默认评分方法，如 R^2）
#     cv=5,       # 交叉验证的折数（默认为 5 折交叉验证）
#     n_jobs=None # 并行计算的 CPU 核数（None 表示默认）
# )
# ```

# In[9]:


degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          param_name='polynomialfeatures__degree',
                                          param_range=degree, cv=5)
# train_score(21,5), 每一行表示一个degree参数的5次交叉验证得分
plt.plot(degree, np.median(train_score, 1), color='blue',
         label='training score')  # 取5次交叉验证的平均值分数
plt.plot(degree, np.median(val_score, 1),
         color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')


# 结论：
# 1. 训练得分一定大于验证得分，复杂度越高，训练得分上升， 但验证得分会由于过拟合骤降。
# 2. 3次多项式是 平衡 偏差和方差 最好的点。
#
# 现在来看下长什么样子

# In[14]:


print(X.shape, y.shape)
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)


# 确实，3次多项式拟合效果很好！

# ### 5.3.3　学习曲线
# 影响模型复杂度的另一个重要因素是最优模型往往受到训练数据量的影响

# In[21]:


X2, y2 = make_data(200)  # 生成了5倍的数据degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2,
                                            param_name='polynomialfeatures__degree',
                                            param_range=degree,  cv=7)

# 200数据的验证曲线
plt.plot(degree, np.median(train_score2, 1), color='blue',
         label='training score')
plt.plot(degree, np.median(val_score2, 1),
         color='red', label='validation score')
# 50小数据的验证曲线: 虚线
plt.plot(degree, np.median(train_score, 1), color='blue', alpha=0.3,
         linestyle='dashed')
plt.plot(degree, np.median(val_score, 1), color='red', alpha=0.3,
         linestyle='dashed')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.scatter(X2.ravel(), y2)


# 结论：
# 1. 大数据支持更高的复杂多项式，而小数据在维度13左右就过拟合骤降了。
#
# 这种通过扩大数据规模对比模型得分， 称为学习曲线。
#
# 因此，**学习曲线反映 数据量 对模型影响。** **（验证曲线反映 超参数（复杂度）对 模型得分的影响）。**
#
# 模型得分是平衡训练得分和验证得分。
#
# 学习曲线最最重要的就是，随着样本的增加，模型得分一定会收敛，即再多样也没有用

# ![图片.png](attachment:41572382-d8e4-4e72-8cc0-0d1049591428.png)

# > 因此，即模型能力用尽了， 只能换模型才能了

# 绘制学习曲线

# ---
# | 参数           | 类型                 | 说明 |
# |--------------|------------------|------|
# | `estimator`  | 估计器对象        | 需要评估的机器学习模型（如 `SVC()`、`RandomForestClassifier()`）。 |
# | `X`          | 数组或矩阵        | 训练数据的特征矩阵。 |
# | `y`          | 数组              | 训练数据的目标变量（标签）。 |
# | `train_sizes` | 数组 (默认 5 个点) | 训练集的不同子集大小（如 `np.linspace(0.1, 1.0, 5)` 代表 10% 到 100% ）。 |
# | `cv`         | int / 交叉验证策略 | 交叉验证的折数或交叉验证方法（如 `5` 或 `KFold(n_splits=5)`）。 |
# | `scoring`    | str / callable   | 评估指标（如 `"accuracy"`、`"neg_mean_squared_error"`）。 |
# | `n_jobs`     | int             | 并行计算的作业数（`-1` 使用所有 CPU 核心）。 |
# | `shuffle`    | bool            | 是否在划分数据前进行洗牌（默认 `False`）。 |
# | `random_state` | int / None     | 随机种子（适用于 `shuffle=True` 时）。 |
# | `verbose`    | int             | 详细程度（`0` 为不输出信息）。 |
# | `return_times` | bool           | 是否返回训练和测试的执行时间（默认为 `False`）。 |
# ---
#
#
#

# In[25]:


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
                                         X, y, cv=7,
                                         train_sizes=np.linspace(0.3, 1, 25))  # 训练集大小，分为25份 从30%到1

    print(N, train_lc.shape, val_lc.shape)  # 每行位一个数据量在交叉验证的得分，N训练样本的实际数量。
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray',
                 linestyle='dashed')  # 绘制水平线（horizontal lines）

    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')


# **结论：**
# 1. 训练得分与验证得分一定会收敛随着数据量变化。
# 2. 高维度即高复杂度的模型需要更多的数据量，相对低纬度而言

# ### 5.3.4　验证实践：网格搜索
# 实际中，模型会有多个超参数，学习曲线和验证曲线会变为多维曲面。 这时，找出最优得分点 不容易！
#
# grid_search 工具🆗

# In[29]:


# 多个超参数：维度，是否截距
param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)


# In[31]:


grid.best_params_  # 获得最优超参数


# In[1]:


model = grid.best_estimator_

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)


# In[ ]:


# ## 5.4　特征工程
# 就是清理干净数据，变为特征矩阵。

# ### 5.4.1　分类特征

# 把分类标签01化，独热编码。

# In[10]:


data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]


# DictVectorizer自动把 所有字符串标签转化为One-hot， 但维度骤增！

# In[5]:


vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)


# In[9]:


vec.get_feature_names_out()  # 得到每个拆分后的解释！


# 由于太多0通过稀疏矩阵存储1

# In[12]:


vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)


# ### 5.4.2　文本特征
# 把一串文本转换为一组数值。
#
#

# 最简单的方式：就是单词统计，单词为列，一行显示了由哪几个单词组成
#

# In[13]:


sample = ['problem of evil',
          'evil queen',
          'horizon problem']


# In[14]:


vec = CountVectorizer()
X = vec.fit_transform(sample)
X  # 稀疏矩阵


# In[17]:


pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())


# 缺点就是常用词权重太高了，显然影响分类效果。
#
# 通过 TF–IDF（term frequency–inverse document frequency，词频逆文档频率），通过单词在文档中出现的频率来衡量其权重 。
# 1. TF: 出现频率高越重要 （在当前文档）
# 2. IDF: 但是一些常用词频率过高 但没有信息，反而一些少量词汇 更重要，如 量子。（在很多文档）

# In[19]:


vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())


# ### 5.4.3　图像特征
# 在 Scikit-Image 项目

# ### 5.4.4　衍生特征
# 是通过数学变换衍生出来的，并非是原数据字段。
# 变换输入。

# In[20]:


x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)


# In[25]:


X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)


# 显然 这些数据不能通过直线拟合

# 添加了 平方，立方，。 即将一次多项式转换为3次多项式拟合。
# 特征扩大到了3

# In[24]:


poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)


# In[27]:


model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)


# **思想！：不改变模型，而是改变输入，扩充输入， 提高效果**。 常称为基函数回归

# ### 5.4.5　缺失值填充

# In[30]:


X = np.array([[nan, 0,   3],
              [3,   7,   9],
              [3,   5,   2],
              [4,   nan, 6],
              [8,   8,   1]])
y = np.array([14, 16, -1,  8, -5])


# In[34]:


imp = SimpleImputer(strategy='mean')  # 列均值填充
X2 = imp.fit_transform(X)
X2


# In[36]:


model = LinearRegression().fit(X2, y)
model.predict(X2)


# ### 5.4.6　特征管道
# 将上述方法连在一起处理时候可以简化，比如：
# 1. 均值填充缺失
# 2. 衍生特征到二次
# 3. 拟合线性回归
#
# 通过管道pipeline对象

# In[40]:


model = make_pipeline(SimpleImputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())
model.fit(X, y)
y


# ## 5.5 朴素贝叶斯
# 简单的根源在于，根据数据直接计算，不需要学习迭代

# In[ ]:


data = fetch_20newsgroups()
data.head()


# ### 5.5.1　贝叶斯分类

# In[1]:


sns.set()


# ### 5.5.2　高斯朴素贝叶斯
# 假设每个数据（x,y）服从高斯分布，即知道模型但不知道复杂度参数：均值、方差。
#
# 高斯分布在高斯朴素贝叶斯中只是一个假设，假设每个类别的特征数据服从高斯分布，即似然函数。这个假设简化了模型，使得我们能够使用简单的 **均值和方差** 来估计每个类别的特征分布。
#
#
# 在进行 **参数估计** 时，贝叶斯估计和极大似然估计只是通过数据来估算模型参数（均值、方差等），而不需要在估计过程中明确关注整个高斯分布模型。简而言之，**你需要关注的是如何根据数据推断出合适的参数，而不需要关心模型的具体形式**。

# ---
# > 极大似然估计（MLE） vs. 贝叶斯估计（MAP） 在高斯朴素贝叶斯中的区别
#
# 我们用一个简单的例子来说明 **MLE 和 贝叶斯估计（MAP）** 在 **高斯朴素贝叶斯** 中的 **参数计算区别**。
#
#
# 假设：我们有一个简单的二分类数据集
# 我们观察到了一些样本，每个样本只有一个特征 `x`，类别 `y` 只有两种：**类别 0** 和 **类别 1**。
#
#  数据如下：
#
# | `x`  | `y` |
# |------|-----|
# | 2.0  | 0   |
# | 2.2  | 0   |
# | 1.8  | 0   |
# | 3.0  | 1   |
# | 3.2  | 1   |
# | 2.8  | 1   |
#
# 任务：用 **高斯分布** 来估计每个类别的均值 `μ` 和方差 `σ^2`。
#
# 方法 1：极大似然估计（MLE）
# 极大似然估计直接用 **训练数据计算均值和方差**：
#
# - 对于 **类别 0**：
#   $$
#   \hat{\mu}_0 = \frac{2.0 + 2.2 + 1.8}{3} = 2.0
#   $$
#
#   $$
#   \hat{\sigma}_0^2 = \frac{(2.0 - 2.0)^2 + (2.2 - 2.0)^2 + (1.8 - 2.0)^2}{3} = \frac{0 + 0.04 + 0.04}{3} = 0.0267
#   $$
#
# - 对于 **类别 1**：
#   $$
#   \hat{\mu}_1 = \frac{3.0 + 3.2 + 2.8}{3} = 3.0
#   $$
#
#   $$
#   \hat{\sigma}_1^2 = \frac{(3.0 - 3.0)^2 + (3.2 - 3.0)^2 + (2.8 - 3.0)^2}{3} = \frac{0 + 0.04 + 0.04}{3} = 0.0267
#   $$
#
# **特点**：
# - ✅ **MLE** 只依赖于 **已有数据**，没有任何先验信息。
# - ⚠️ **当数据量少时，计算的方差可能会偏小，容易过拟合**。
#
#
# 方法 2：贝叶斯估计（MAP）
# 贝叶斯估计会在方差计算时 **加一个平滑项**，避免数据太少导致计算出的方差过小：
#
# $$
# \hat{\sigma}^2 = \frac{1}{N} \sum (x - \mu)^2 + \lambda
# $$
#
# 比如，我们加一个 **小的平滑项** `λ = 0.01`：
#
# - **类别 0**：
#   $$
#   \hat{\sigma}_0^2 = 0.0267 + 0.01 = 0.0367
#   $$
#
# - **类别 1**：
#   $$
#   \hat{\sigma}_1^2 = 0.0267 + 0.01 = 0.0367
#   $$
#
# **特点**：
# - ✅ 避免了方差过小的问题，使得模型在数据少时更稳定。
# - ⚠️ 需要选择一个合适的平滑参数 `λ`。
#
#
# 核心区别总结
#
# |  方法  | 计算均值 `μ`  | 计算方差 `σ^2` | 适用场景 |
# |--------|------------------|------------------|-----------|
# | **MLE** | 直接用数据计算  | $σ^2 = \frac{1}{N} \sum (x - \mu)^2$ | 数据量大时效果好，易过拟合 |
# | **MAP** | 直接用数据计算  | $σ^2 = \frac{1}{N} \sum (x - \mu)^2 + \lambda$ | 数据少时更稳定，防止方差过小 |
#
#
# 总结
#
# - **MLE** 是纯数据驱动的，适合大数据量场景，但可能会导致方差过小，容易过拟合。
# - **MAP** 在小数据集下更稳定，因为它加入了一个平滑项来防止方差过小。
#

# ---

# #### 模型学习

# In[ ]:


# In[6]:


X, y = make_blobs(n_samples=100, n_features=2, centers=2,
                  random_state=2, cluster_std=1.5)
print(X.shape, y.shape)


# In[10]:


model = GaussianNB()  # 默认使用最大似然估计
model.fit(X, y)

print("每个类别的均值:", model.theta_)  # [n_classes, n_features] 每一行表示一个类别 各个特征均值
print("每个类别的方差:", model.var_)  # [n_classes, n_features] 每一行表示一个类别 各个特征方差。 特征独立


# #### 预测可视化📊

# In[12]:


rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)


# 💡可以看到，高斯朴素贝叶斯的有一条分界线，这条线常常是二次方曲线

# #### 🔑可视化高斯分布
# 结论：高斯分布衰减极为迅速。

# In[9]:


# 获取每个类别的均值和方差
means = model.theta_
covariances = model.var_

# 设定网格范围
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# 创建图像
plt.figure(figsize=(8, 6))

# 绘制每个类别的高斯分布等高线
for i in range(len(means)):
    # 创建一个二维正态分布
    rv = multivariate_normal(mean=means[i], cov=np.diag(
        covariances[i]))  # np.diag创建对角线矩阵
    # 计算概率密度函数的值
    Z = rv.pdf(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z.max())
    # 绘制等高线
    contours = plt.contour(xx, yy, Z, levels=5, cmap="Blues", alpha=0.6)
    # 为等高线添加标签
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
plt.title('Gaussian Naive Bayes - Gaussian Distribution Contours')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()


#

# ### 5.5.3 多项式朴素贝叶斯

# #### 1. 示例：新闻分类

# In[17]:


data = fetch_20newsgroups()
data.target_names


# In[18]:


categories = data.target_names
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])


# **模型学习**

# In[19]:


model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[20]:


model.fit(train.data, train.target)
labels = model.predict(test.data)


# In[21]:


mat = confusion_matrix(test.target, labels)


# In[22]:


sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[23]:


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


predict_category('sending a payload to the ISS')


# #### 2. 示例： 今日头条新闻分类

# In[128]:


f = open('toutiao_cat_data.txt', encoding='utf8')
data = f.readlines()
data = [line.split('_!_') for line in data]
data = pd.DataFrame(
    data, columns=['新闻ID', '分类code', '分类名称', '新闻字符串（仅含标题）', '新闻关键词'])
data.info()


# In[129]:


data.describe()


# In[130]:


data.head()


# In[131]:


print(data[data['新闻ID'] == '6554616677084955139'])


# **1. 数据清理**
# - 只匹配，中文字符、英文字符和数字
# - 中英文符合统一

# In[132]:


data['新闻'] = data['新闻关键词'] + '' + data['新闻字符串（仅含标题）']


# In[133]:


symbol_map = str.maketrans({
    '!': '！',
    '?': '？',
    ',': '，',
    '.': '。',
    ':': '：',
    ';': '；',
    "'": '’',
    '"': '“',
    '(': '（',
    ')': '）',
    '-': '——',
    '_': '＿'
    # 添加其他符号的映射
})
data['新闻'] = data['新闻'].str.translate(symbol_map)  # translate集体映射替换


# In[134]:


data['新闻'].tail()


# In[135]:


data['新闻'] = data['新闻'].str.replace('\n', '')
data['新闻'].head()


# In[136]:


data['新闻长度'] = data['新闻'].apply(len)
longest_text = data.loc[data['新闻长度'].idxmax()]  # 找到最长的一行
longest_text


# In[137]:


X_news = data['新闻']  # 向量化只能处理一列
Y_news = data['分类名称']
print(X_news.shape, Y_news.shape)


# **2.分割数据集**

# In[138]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_news, Y_news, random_state=1)
print(Xtrain.shape, Ytrain.shape, Xtest.shape)


# **🚀 TF使用空格和，分词， 对中文没有效果。 需要自定义先对中文分词。**

# In[139]:


def jieba_tokenizer(txt):
    return jieba.lcut(txt)


result = jieba_tokenizer('发酵床的垫料种类有哪些？哪种更好？')
print(result)


# **3. 模型学习**

# In[149]:


model = make_pipeline(TfidfVectorizer(
    tokenizer=jieba_tokenizer, token_pattern=None), MultinomialNB(alpha=1))


# In[150]:


model.fit(Xtrain, Ytrain)


# In[151]:


vectorizer = model.named_steps['tfidfvectorizer']
features = vectorizer.get_feature_names_out()
print(type(features))  # 查看分词特征
print(len(features))
print(features[:30])


# **4. 模型预测**

# In[152]:


predicated = model.predict(Xtest)
predicated_sobj = pd.Series(predicated, index=Ytest.index)
mat_df = pd.DataFrame({'预测名称': predicated_sobj, '分类名称': Ytest})


# **5. 准确率**

# In[153]:


accuracy_score(predicated, Ytest)


# **6. 混淆矩阵**

# In[154]:


labels = np.unique(Y_news)
mat = confusion_matrix(Ytest, predicated, labels=labels)
plt.figure(figsize=(10, 8))  # 设置宽度为10，高度为8
sns.heatmap(mat.T, square=True, annot=True, fmt='d',
            xticklabels=labels, yticklabels=labels, cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[148]:


print("Precision:", precision_score(Ytest, predicated, average='weighted'))
print("Recall:", recall_score(Ytest, predicated, average='weighted'))
print("F1 Score:", f1_score(Ytest, predicated, average='weighted'))


# 结论：总是误分类到news_tech

# In[146]:


category_counts = data['分类名称'].value_counts()
print(category_counts)


# In[ ]:


# ## 5.6　专题：线性回归
#

# > 如果说朴素贝叶斯是解决分类任务的好起点，那么线性回归模型就是解决回归任务的好起点。拟合速度非常快，而
# 且很容易解释。

# In[3]:


sns.set()


# ### 5.6.1　简单线性回归

# In[11]:


rng = np.random.RandomState(0)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)


# In[15]:


model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)


# In[17]:


print("Model slope:    ", model.coef_[0])  # 接近2
print("Model intercept:", model.intercept_)  # 接近-5


# ### 5.6.2　基函数回归
# 基函数将变量的线性回归转为非线性回归。 仍然是线性模型。 （ 5.3 节和 5.4 节）

# 1. 多项式基函数

# 转变为只是高次回归

# In[18]:


x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)  # 转换器将一维数组转换为3维
poly.fit_transform(x[:, None])


# In[19]:


poly_model = make_pipeline(PolynomialFeatures(7),
                           LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)


# 2. 高斯基函数

# ### 5.6.3　正则化

# In[ ]:


# In[ ]:


# In[ ]:


# ### 5.6.4　案例：预测自行车流量
# 创建一个简单的线性 回归模型来探索与自行车数量相关的天气和其他因素，从而评估任意一种因素对骑车人数 的影响。

# In[ ]:


# In[ ]:


# In[ ]:


# ## 5.7　专题：支持向量机
# 可分类，可回归

# In[ ]:


# In[2]:


sns.set()


# In[3]:


X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')


# 有多条线可以分割，选择一个最好的。

# In[4]:


xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5)


# ### 5.7.2　支持向量机：边界最大化

# In[5]:


xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA',
                     alpha=0.4)

plt.xlim(-1, 3.5)


# 1. 拟合支持向量机

# In[6]:


model = SVC(kernel='linear', C=1E10)
model.fit(X, y)


# In[7]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """画二维SVC的决策函数"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建评估模型的网格
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)  # x坐标集(n,1)， y坐标集(n,1)
    xy = np.vstack([X.ravel(), Y.ravel()]).T  # (n, 2) 的二维数组 xy
    P = model.decision_function(xy).reshape(X.shape)  # 到决策边界的距离

    # 画决策边界和边界
    ax.contour(X, Y, P, colors='k',   levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # 画支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[8]:


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)


# In[9]:


model.support_vectors_


# In[12]:


def plot_svm(N, ax=None):
    X, y = make_blobs(n_samples=N, centers=2,
                      random_state=0, cluster_std=0.60)
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for axi, N in zip(axes, [60, 120]):
    plot_svm(N, axi)
    axi.set_title(f'N = {N}')
plt.show()


# In[11]:


interact(plot_svm, N=[10, 200], ax=fixed(None))


# 2. 超越线性边界：核函数SVM模型
#
# 非线性可分原数据 通过 核函数 线性可分

# In[ ]:


# In[14]:


X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)


# In[15]:


r = np.exp(-(X ** 2).sum(1))


# In[ ]:


# In[18]:


def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')


plot_3D()


# **但通常很难选择基函数，而不是像这么简单**。
# SVC通过核参数设置

# In[21]:


clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, lw=2, facecolors='none', edgecolors='red')  # ✅ 这样一般能正确渲染圈圈


# **3. SVM优化：软化边界**
# 如果你的数据有一些重叠该怎么办呢？

# In[24]:


X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=1.2)  # cluster_std 标准差，控制集群的分散程度
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')


# In[26]:


X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)  # C参数控制能有多少点在边界内
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none',  edgecolors='red')
    axi.set_title('C = {0:.1f}'.format(C), size=14)


# ### 5.7.3　案例：人脸识别
# Wild 数据集中带标记的人脸
#

# In[1]:


faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


# In[4]:


# In[8]:


fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])


# 每个图像包含 [62×47]、接近 3000 像素。虽然可以简单地将每个像素作为一个特征。但通常使用降维，下面用PCA提取150个

# In[9]:


pca = PCA(n_components=150, whiten=True,
          random_state=42, svd_solver='randomized')
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)


# In[13]:


Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)


# **用网格搜索交叉检验来寻找最优参数组合。**
# 参数 C（控制边界线的硬 度）和参数 gamma（控制径向基函数核的大小），

# In[15]:


param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

get_ipython().run_line_magic('time', 'grid.fit(Xtrain, ytrain)')
print(grid.best_params_)


# In[16]:


model = grid.best_estimator_
yfit = model.predict(Xtest)


# In[18]:


fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)


# 分类报告

# In[19]:


print(classification_report(ytest, yfit,
                            target_names=faces.target_names))


# 示例： 聚类

# In[134]:


Xtrain.toarray()


# In[137]:


model = make_pipeline(CountVectorizer(), GaussianMixture(
    n_components=len(labels), covariance_type='full'))


# In[138]:


model.fit(Xtrain)


# https://cloud.tencent.com/developer/ask/sof/111732133

#
