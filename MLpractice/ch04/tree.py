# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 23:26:19 2024

@author: 63517
"""

from math import log
import numpy as np
# 
def calcShanon(dataSet):
    labelsCount = {}
    for line in dataSet:
        label = line[-1]
        labelsCount[label] = labelsCount.get(label, 0) + 1
    shanon = 0.0
    total = len(dataSet)
    for k in labelsCount:
        prob = labelsCount[k] / total
        shanon -= prob * log(prob, 2)
    return shanon

def chooseBestFeature(dataSet):
    hd = calcShanon(dataSet)
    total = dataSet.shape[0]
    max_increase_entropy = 0.0
    bestFeature = 0
    # 计算每种特征的信息增益
    for axis in range(dataSet.shape[1] - 1):
        unique_vals, counts = np.unique(dataSet[:,axis], return_counts=True)
        axis_entropy = 0.0
        for j in range(len(unique_vals)):
            subDataSet = splitDataSet(dataSet, axis, unique_vals[j])
            axis_entropy += counts[j] / total * calcShanon(subDataSet)
        axis_entropy = hd - axis_entropy
        if axis_entropy > max_increase_entropy:
            bestFeature = axis
    return bestFeature


# 按照某个类的特征值提取
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for line in dataSet:
        if line[axis] == value:
            retLine = np.concatenate((line[:axis], line[axis + 1:]))
            retDataSet.append(retLine)
    return np.array(retDataSet)

def major(vector):
    # 返回数组的众数
    unique_vals, counts = np.unique(vector,return_counts=True)
    return unique_vals[np.argmax(counts)]

# 创建树
def createTree(dataSet):
    if dataSet.shape[1] == 1:
        # 已经分完，只剩下label, 多数表决
        return major(np.reshape(dataSet, (1, len(dataSet))))
    if len(set(dataSet[:, -1])) == 1:
        # 数据集具有同样的分类, 返回分类值
        return dataSet[0][-1]
    tree = {}
    feature = chooseBestFeature(dataSet)
    unique_vals = np.unique(dataSet[:,feature])
    for v in unique_vals:
        subDataSet = splitDataSet(dataSet, feature, v)
        tree[v] = createTree(subDataSet)
    return tree
        

def createDataSet():
    np.random.seed(0)
    dataSet = np.random.randint(2, size = (6,5))

    return dataSet
dataSet = createDataSet()
print('dataset', dataSet)
chooseBestFeature(dataSet)

tree=  createTree(dataSet)
print(tree)
