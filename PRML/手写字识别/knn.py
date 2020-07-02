# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 09:06:21 2020

@author: Joash
"""


import numpy as np

def knn(testSet,trainSet,labels,k):
    dist=(((trainSet-testSet)**2).sum(1))**0.5
    sortedDist=dist.argsort()
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDist[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    maxType=0
    maxCount=-1
    for key,value in classCount.items():
        if value>maxCount:
            maxType=key
            maxCount=value
    return maxType