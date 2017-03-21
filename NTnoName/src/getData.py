# -*- coding:utf-8 -*-
def DataFilter(fname1,fname2):
    f1=open(fname1)
    f2=open(fname2,'w')
    while 1:
        line=f1.readline()
        if not line:
            break
        #把每个token用空格分开，这里主要处理{}
        line=line.replace('{',' { ')
        line=line.replace('}',' } ')
        line=' '.join(line.split())
        code=line.split(' ')
        if len(code)>=11:
            f2.write(line)
            f2.write('\n')
    f1.close()
    f2.close()

DataFilter('D:/py_project/Tensorflow/s-lstm/data/res-3.8W-new.txt','D:/py_project/Tensorflow/s-lstm/data/codeGT10.txt')

def SplitData(fname):
    f=open(fname)
    trainfile=open('D:/py_project/Tensorflow/s-lstm/data/train.txt','w')
    testfile=open('D:/py_project/Tensorflow/s-lstm/data/test.txt','w')
    code=f.read().split('\n')
    splitIndex=int(0.7*len(code))
    trainData=code[:splitIndex]
    testData=code[splitIndex:]

    for i in range(len(trainData)):
        trainfile.write(trainData[i])
        trainfile.write('\n')
    for i in range(len(testData)):
        testfile.write(testData[i])
        testfile.write('\n')
    f.close()
    trainfile.close()
    testfile.close()

SplitData('D:/py_project/Tensorflow/s-lstm/data/XXX.txt')





