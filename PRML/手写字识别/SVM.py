# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:44:44 2020

@author: Joash
"""



import numpy as np 
from os import listdir
from sklearn import svm


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

hwLabels = []
trainingFileList = listdir('trainingDigits') #读取文件夹里的文件名，一维数组

m = len(trainingFileList)
trainingMat = np.zeros((m,1024))
for i in range(m):
    fileNameStr = trainingFileList[i] #提取文件名
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

model = svm.SVC()
model.fit(trainingMat,hwLabels)

testFileList = listdir('testDigits')
errorCount = 0.0
numTest = len(testFileList)
for i in range(numTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) #读取测试数据
    classifierResult = model.predict(vectorUnderTest)
    print("SVM得到的辨识结果是：%d，实际值是：%d" % (classifierResult,classNumStr))
    if (classifierResult != classNumStr):
        errorCount += 1.0

print("\n辨识错误数量为：%d" % errorCount)

print("\n辨识率为：%f %%" %((1-errorCount/float(numTest))*100)) #用%%代替%
#print("\n辨识率为: %f ％" % ((1-errorCount/float(numTest))*100))#参考代码