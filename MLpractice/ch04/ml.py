# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:07:47 2024

@author: 63517
"""

import knn
mat,labels = knn.file2matrix('datingTestSet.txt')

normMat = knn.autoNorm(mat)