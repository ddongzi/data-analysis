# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:52:13 2024

@author: 63517
"""

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = 'sawtooth', fc='0.8')
leafNode = dict(
    boxstyle="round,pad=0.5",  # 使用带 padding 的圆角矩形
    fc="lightgreen",           # 填充颜色为浅绿色
    ec="black",                # 边框颜色为黑色
    lw=2,                      # 边框宽度为2
    alpha=0.9,                 # 透明度
    linestyle="--"             # 边框线条样式为虚线
)
arrow_args = dict(arrowstyle = '<-')

def plotNode(nodeText, parentLoc, textLoc, nodeType):
    plt.annotate(nodeText, xy = parentLoc, bbox= nodeType, xytext=textLoc, arrowprops=arrow_args)

def getNumLeafs(tree):
    numLeafs = 0
    for k in tree.keys():
        if not isinstance(tree[k], dict):
            # 是叶子节点
            numLeafs += 1
        else:
            numLeafs += getNumLeafs(tree[k])
    return numLeafs

def getTreeDeepth(tree):
    depth = 0
    maxDepth = 0
    for k in tree.keys():
        if not isinstance(tree[k], dict):
            # 是叶子节点
            depth = 1
        else:
            depth += getTreeDeepth(tree[k])
    maxDepth = max(depth,maxDepth)
    return maxDepth

def plotTree(tree, parentLoc,textLoc):
    # 绘制根节点
    root = list(tree.keys())[0]
    
    plotNode(root, parentLoc, textLoc, decisionNode)
    
    # 绘制子节点
    parentLoc = textLoc
    keyslist = list(tree.keys())
    
    keysoffset = [ int(i-len(keyslist)//2)  for i in range(len(keyslist))]

    for i in range(len(keyslist)):
        k = keyslist[i]

        childTextLoc = (textLoc[0] + keysoffset[i]* 0.2, textLoc[1] - 0.2 )
        if not isinstance(tree[k], dict):
            # 是叶子节点
            plotNode(tree[k], parentLoc, childTextLoc, leafNode)
        else:
            plotTree(tree[k], parentLoc, childTextLoc)
tree = {
    'Outlook': {
        'Sunny': {
            'Humidity': {
                'High': 'No',
                'Normal': 'Yes'
            }
        },
        'Overcast': 'Yes',
        'Rain': {
            'Wind': {
                'Strong': 'No',
                'Weak': 'Yes'
            }
        }
    }
}


fig= plt.figure()
plt.axes()

# 绘制树根节点
rootLoc = (0.5, 1)
plt.text(rootLoc[0],rootLoc[1], 'Fake root' )
textLoc = (rootLoc[0], rootLoc[1]- 0.2)
plotTree(tree, rootLoc, textLoc)