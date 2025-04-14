#!/usr/bin/env python
# coding: utf-8

# [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

# ## 1.1. Linear Models
# $\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p$
# `w = (w_1,..., w_p)` as ``coef_`` and `w_0` as ``intercept_``.

# ### 1.1.1. Ordinary Least Squares
# - $\min_{w} || X w - y||_2^2$
# - The coefficient estimates for Ordinary Least Squares rely on the independence of the features.

# In[1]:


from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]  # only one feature
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=20, shuffle=False)
reg = LinearRegression().fit(X_train, y_train)


# In[2]:


y_pred = reg.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")


# In[3]:


fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].scatter(X_train, y_train, label="Train data points")
ax[0].plot(X_train, reg.predict(X_train), linewidth=2,
           color='orange', label='model:predict')
ax[0].set(xlabel='feature', ylabel='target', title='train set')
ax[0].legend()

ax[1].scatter(X_test, y_test, label="Test data points")
ax[1].plot(X_test, reg.predict(X_test), linewidth=2,
           color='orange', label='model:predict')
ax[1].set(xlabel='feature', ylabel='target', title='test set')
ax[1].legend()


# #### 1.1.1.1. Non-Negative Least Squares
# It is possible to constrain all the coefficients to be non-negative

# In[4]:


np.random.seed(42)
n_samples, n_features = 200, 50
X = np.random.randn(n_samples, n_features)
coef = np.random.randn(n_features)
coef[coef < 0] = 0
y = np.dot(X, coef) + np.random.normal(size=(n_samples, ))

X_train, X_test, y_train, y_test = train_test_split(X, y)
reg_nn = LinearRegression(positive=True)
y_pred_nn = reg_nn.fit(X_train, y_train).predict(X_test)
print(f"reg non-negtive r2 score {r2_score(y_test, y_pred_nn)}")

reg = LinearRegression()
y_pred = reg.fit(X_train, y_train).predict(X_test)
print(f"reg r2 score {r2_score(y_test, y_pred)}")


# In[5]:


print(f'reg coef: {reg.coef_}')
print(f'reg non-negtive coef: {reg_nn.coef_}')


# In[6]:


fig, ax = plt.subplots()
ax.plot(reg.coef_, reg_nn.coef_, linewidth=0, marker='.')
low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()
low = min(low_x, low_y)
high = max(high_x, high_y)
ax.plot([low, high], [low, high])
ax.set_xlabel('reg coef')
ax.set_ylabel('reg positive coef')


# ### 1.1.2. Ridge regression and classification
#

# #### 1.1.2.1. Regression
# Ridge regression addresses some of the problems of Ordinary Least Squares by **imposing a penalty on the size of the coefficients (L2)**. The ridge coefficients minimize a penalized residual sum of squares:
# $$
#    \min_{w} || X w - y||_2^2 + \alpha ||w||_2^2
# $$
# The complexity parameter $\alpha \geq 0$ controls the amount of shrinkage: the larger the value of $\alpha$, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

# ---

# > $\alpha$

# In[7]:


# X is the 10x10 Hilbert matrix
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)  # 对数尺度上均匀分布，而非等线性间隔
coef_ = []
for a in alphas:
    ridge = linear_model.Ridge(fit_intercept=False, alpha=a)
    ridge.fit(X, y)
    coef_.append(ridge.coef_)
# print(coef_[0])
ax = plt.gca()
# (200,) (200, 10)
ax.plot(alphas, coef_)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title("Ridge coefficients as a function of the regularization")
plt.axis('tight')
plt.show()


# 📉
# - Since coef_ has 10 columns, the plot will contain 10 curves, each representing the change in a feature's coefficient as alpha increases.
# - If $\alpha$ is too large, most coefficients will be 0, meaning this model only a constant value.
#
# ---

# In[8]:


reg = linear_model.Ridge(alpha=0.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])
print('coef_ : ', reg.coef_)
print('intercept : ', reg.intercept_)


# Note that the class Ridge allows for the user to specify that the solver be automatically chosen by setting solver="auto". Solver determine how the Ridge regression **compute the optimal weights $w$**, depending on the dataset size, sparsity, and numerical efficiency.

# #### 1.1.2.2. Classification
# The Ridge regressor has a classifier variant: *RidgeClassifier*. This classifier first converts binary targets to **{-1, 1}** and then treats the problem as a regression task, optimizing the same objective as above. For multiclass classification, the problem is treated as **multi-output regression**, and the predicted class corresponds to the output with the highest value.
#
# while the least squares loss is **not commonly used for classification** tasks, it can still give similar results as more traditional approaches like logistic regression or SVM. Additionally, the RidgeClassifier's use of this loss function provides flexibility in choosing solvers, which can lead to **more efficient computations** in certain situations.
#
# The **RidgeClassifier** can be significantly faster than **LogisticRegression** in multi-class problems because it only needs to compute the ❓projection matrix once, regardless of how many classes there are, while **LogisticRegression** needs to perform additional computations for each class. This gives the **RidgeClassifier** an advantage in terms of speed, particularly when working with a large number of classes.

# ---
#

# [📌Classification of text documents using sparse features](./examples/Working%20with%20text%20documents.ipynb#classification-of-text-documents-using-sparse-features)
#

#

# #### 1.1.2.3 complexity

# #### 1.1.2.4 Set the regularization param:  RidgeCV , RidgeClassifierCV
# they have built-in cv of the alpha. This work in the same way as GridSearchCV

# In[11]:


# In[13]:


np.logspace(-6, 6, 13)  # 13个


# In[18]:


reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])


# In[19]:


reg.alpha_


# In[ ]:


# ### 1.1.3 Lasso
#

# Compare to Ridge, Lasso have a L1 penality . When we need more 0 coefficients (sparse model), we can do it
# $$
#    \min_{w} || X w - y||_2^2 + \alpha ||w||_1
# $$

# ---
# > L1 VS L2
# 1. 定义
# L1: 某些系数会被完全压缩到 0，从而进行特征选择。
# L2: 不会让系数变成 0，而是让所有系数向 0 方向收缩，类似于“均匀压缩”。
# 2. 直观理解（几何角度）
# Lasso 和 Ridge 的不同可以用约束区域来解释：
# L1：$i.e. (|w_1| + |w_2| \leq C)$约束是一个 **菱形（L1 球）**，优化时解往坐标轴方向靠近，导致某些系数精确为 0，实现变量选择。
# L2: $i.e. (w_1^2 + w_2^2 \leq C^2)$约束是一个 **圆形区域**，优化过程中梯度会均匀收缩所有系数，导致所有特征都保留，只是值变小了。
# ---

# In[1]:


# 生成 w1 和 w2 的网格
w1 = np.linspace(-1.5, 1.5, 400)
w2 = np.linspace(-1.5, 1.5, 400)
W1, W2 = np.meshgrid(w1, w2)

# MSE 等高线（假设某个二元回归问题的等值线）
Z = W1**2 + 2*W1*W2 + 1.5*W2**2  # 这是一个典型的二次损失函数

# 创建绘图区域
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# --- L1 正则化（Lasso）：菱形约束 ---
ax[0].contour(W1, W2, Z, levels=10, cmap='coolwarm')  # 绘制损失函数等高线
ax[0].plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1], 'g-',
           linewidth=2, label="L1 Constraint")  # 画菱形（L1 范数约束）
ax[0].set_title("L1 Regularization (Lasso)")
ax[0].set_xlabel(r'$w_1$')
ax[0].set_ylabel(r'$w_2$')
ax[0].axhline(0, color='black', linewidth=0.5)
ax[0].axvline(0, color='black', linewidth=0.5)
ax[0].legend()

# --- L2 正则化（Ridge）：圆形约束 ---
ax[1].contour(W1, W2, Z, levels=10, cmap='coolwarm')  # 绘制损失函数等高线
circle = plt.Circle((0, 0), 1, color='g', fill=False,
                    linewidth=2, label="L2 Constraint")  # 画圆形（L2 范数约束）
ax[1].add_patch(circle)
ax[1].set_title("L2 Regularization (Ridge)")
ax[1].set_xlabel(r'$w_1$')
ax[1].set_ylabel(r'$w_2$')
ax[1].axhline(0, color='black', linewidth=0.5)
ax[1].axvline(0, color='black', linewidth=0.5)
ax[1].set_xlim([-1.5, 1.5])
ax[1].set_ylim([-1.5, 1.5])
ax[1].legend()

# 显示图像
plt.tight_layout()
plt.show()


# In[3]:


model = linear_model.Lasso(alpha=0.1)
model.fit([[0, 0], [1, 1]], [0, 1])


# In[5]:


model.predict([[0.5, 0.5]])


# In[ ]:
