# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 00:10:14 2024

@author: 63517
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 50)
plt.plot(x, np.sin(x - 1), color = 'red', linestyle = '--', label='sin(x)')
plt.plot(x, np.cos(x), label = 'cos(x)')

plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

plt.axis('tight')

plt.title('sin')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.legend()

plt.plot(x, np.sin(x), '-ok')


# scatter 精确控制散点
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.randn(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes,alpha=0.3, cmap='viridis')
plt.colorbar() # 显示颜色条

print(x)

# 误差
x = np.linspace(0, 10,50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt ='k') # 


def f(x,y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
x = np.linspace(0, 5, 30)
y = np.linspace(0, 5, 40)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)
plt.contour(X,Y,Z, cmap = 'RdGy')

# 直方图
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)
kwargs = dict(histtype='stepfilled', alpha=0.3,  bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);

# 二维直方
mean = [0,0]
cov = [[1,1], [1,2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T 
plt.hexbin(x, y, gridsize = 30, cmap='Blues')
cb = plt.colorbar(label = 'hexbin')

# 图例
x = np.linspace(0, 10,100)
plt.plot(x, np.sin(x), '-b', label = 'Sin')
plt.plot(x, np.cos(x), '--r', label = 'Sin')

plt.legend()

# 图例不同点
import pandas as pd
cities = pd.read_csv('california_cities.csv')

lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

plt.scatter(lon, lat, label = None,
            c = np.log10(population),cmap='viridis',
            s = area, linewidth = 0, alpha= 0.5)
plt.xlabel('longtitude')
plt.ylabel('latitude')
plt.colorbar(label = 'log$_{10}$(population)')


for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False,labelspacing=1, title='City Area')
plt.title('California Cities: Area and Population');




# 子图
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

for i in range(1,7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2,3,i)), fontsize = 18)

plt.subplots(2,3,sharex='col',sharey='row')

grid = plt.GridSpec(2, 4, wspace=0.4, hspace= 0.3)
plt.subplot(grid[0,0])
plt.subplot(grid[0,1:])
plt.subplot(grid[1,:2])
plt.subplot(grid[1,2:])

# 创建一些正态分布数据
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

plt.GridSpec(4, 4, hspace= 0.2, wspace= 0.2)
main_ax = plt.subplot(grid[:-1, 1:])
y_hist = plt.subplot(grid[:-1, 0], xticklabels = [] ,sharey = main_ax)
x_hist = plt.subplot(grid[-1, 1:],yticklabels = [], sharex = main_ax)

main_ax.scatter(x, y, alpha=0.3, cmap='viridis')
x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical',color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal',color='gray')
y_hist.invert_xaxis()

# 人口变化趋势
births = pd.read_csv('births.csv')
quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.index = [pd.to_datetime(f"2012-{month}-{day}") for (month, day) in births_by_date.index]
fig, ax = plt.subplots(figsize = (12,4))
births_by_date.plot(ax=ax)

style = dict(size = 10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)
ax.set(title = 'usa births by day of year (1969-1999)', ylabel = 'av')

import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%h'));

# 标注
fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')








