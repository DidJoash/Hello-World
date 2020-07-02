# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:47:55 2020

@author: Joash
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def file2matrix(filename):
    fr = open(filename)
    numberOfline = len(fr.readlines())
    returnMat = np.zeros((numberOfline,4))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(",")
        returnMat[index,:] = listFromLine[0:4]
        if listFromLine[-1] == 'Iris-setosa':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'Iris-versicolor':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'Iris-virginica':
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    normDataSet = np.zeros(dataSet.shape)
    normDataSet = (dataSet-minVals)/(maxVals-minVals)
    return normDataSet


flowerDataMat,flowerLabelVector = file2matrix('iris_dataset.txt')
# print(flowerDataMat)
# print(flowerLabelVector)
plt.scatter(flowerDataMat[:,0],flowerDataMat[:,2],c=flowerLabelVector)

dataSet = autoNorm(flowerDataMat)
print(dataSet)

m = 0.8
dataSize = dataSet.shape[0]      
print(dataSize)
trainSize = int(m*dataSize)
testSize = int((1-m)*dataSize)
print(trainSize,testSize)

model = svm.SVC()
model.fit(dataSet[0:trainSize,:],flowerLabelVector[0:trainSize])


error = 0
for i in range(testSize):
    result = model.predict(dataSet[trainSize+i-1,:].reshape(1,-1))
    if result != flowerLabelVector[trainSize+i-1]:
        error += 1
        
print("error=",error/testSize)