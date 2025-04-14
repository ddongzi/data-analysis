# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:03:24 2024
@description : 机器学习实战-KNN分类
@author: 63517
"""

from numpy import *
import operator

import os
import re

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    lines_len = len(lines)
    resMat = zeros((lines_len,3))
    classLableVector = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        resMat[index, :] = listFromLine[0:3]
        classLableVector.append(listFromLine[-1])
        index+=1
    return resMat, array(classLableVector)

def createDataSet():
    group = array([
        [1.0, 1.1],
                      [1.2, 1.0],
                      [0, 0],
                      [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 规一化
def autoNorm(mat):
    minvals = mat.min(0)
    maxvals = mat.max(0)
    ranges = maxvals - minvals
    normMat = zeros(mat.shape)
    m = mat.shape[0]
    normMat = mat - tile(minvals, (m,1))
    normMat = normMat / tile(ranges, (m,1))
    return normMat, ranges, minvals

# intX: 待分类。 dataset: 训练集特征， labels： 标签， k：k近邻
def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance ** 0.5
    sortedDistIndx = distances.argsort()
    
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndx[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key= lambda x:x[1])
    return sortedClassCount[0][0]
# 约会数据分类
def datingClassTest():
    mat, labels = file2matrix('datingTestSet.txt')
    normMat,ranges,minvals = autoNorm(mat)
    m = normMat.shape[0]  
    print(normMat.shape, labels.shape)
    ratio = 0.9
    ts = int(m * ratio)
    test_data = normMat[ts:]
    test_labels = labels[ts:]
    train_data = normMat[0:ts,]   
    train_labels = labels[0:ts,]   
    print(test_data.shape, train_data.shape)
    
    right = 0
    for i in range(test_data.shape[0]):
        # 对每个输入进行分类预测
        predict = classify0(test_data[i], train_data, train_labels, 3)
        print(i, predict, test_labels[i])
        if predict == test_labels[i]:
            right += 1
    print('正确率', right / test_data.shape[0])

def img2vector(filename):
    f = open(filename)
    resList = []
    for i in range(32):
        resList += [int(c) for c in f.readline().rstrip()]
    res = array(resList)
    return res
# 手写体分类
def handWritingClassTest():
    files = [f for f in os.listdir('digits/trainingDigits')]
    train_data = []
    train_labels = []
    for file in files:
        numbers = re.findall(r'\d+', file)
        train_data.append(img2vector('digits/trainingDigits/' + file)) 
        train_labels.append(numbers[0])
    train_data = array(train_data)
    train_labels = array(train_labels)
    print(train_data.shape, train_labels.shape)
    
    files = [f for f in os.listdir('digits/testDigits')]
    test_data = []
    test_labels = []
    for file in files:
        numbers = re.findall(r'\d+', file)
        test_data.append(img2vector('digits/testDigits/' + file)) 
        test_labels.append(numbers[0])
    test_data = array(train_data)
    test_labels = array(train_labels)
    print(test_data.shape, test_labels.shape)
    
    right = 0
    for i in range(test_data.shape[0]):
        predict = classify0(test_data[i], train_data, train_labels, 3)
        if predict == test_labels[i]:
            right += 1
    print('right', right / test_data.shape[0])


# kd树

    
