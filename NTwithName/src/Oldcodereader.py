# -*- coding:utf-8 -*-
import tokenize
import os
# import queue
# f = open('data/code/train.txt', 'r')
# code=f.readline()
# print(code)
# code=code.replace(r'\n','\n')
#
#
# try:
#     tokens=list(tokenize._my_tokenize(code,'utf-8'))
#     for token in tokens:
#         print('%-20s %-20r'%(tokenize.tok_name[token.type],token.string))
# except :
#     print('error')

import tensorflow as tf
import collections

#读取文件，将token以list形式保存
def _read_words(filename):
    with tf.gfile.GFile(filename,'r') as f:
        # return f.read().decode('utf-8').replace("\n","<eos>").split()
        return f.read().decode('utf-8').split(' ')
#
# data=_read_words('data/code/trainLineGT10.txt')
# f=open('data/code/tmp.txt','w')
# f.write(str(data))

#创建词汇表，将token按出现频率排序，转换为word:id
#todo 必须要考虑变量名的问题  否则UNK占比太高
def _build_vocab(filename,n=9999):
    data = _read_words(filename)
    # print(data)
    counter = collections.Counter(data)
    # print(counter[''])
    # print(sum(counter.values()))
    # a=sum(counter.values())
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, values = list(zip(*count_pairs))
    # print(len(values))
    # print(sum(values[0:9999]))
    # b=sum(values[0:9999])
    # print(b*1.0/a)

    #取频率较高的n个word
    words=words[0:n]

    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['UNK']=len(words)
    return word_to_id

#将文件中的token转换为id
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    wordId=[]
    for word in data:
        if word in word_to_id:
            wordId.append(word_to_id[word])
        else:
            wordId.append(word_to_id['UNK'])
    # return [word_to_id[word] for word in data if word in word_to_id]
    return wordId

# #得到token id  10为endmarker
# voc=_build_vocab('data/code/trainGT10.txt')
# # print(len(voc))
# print(voc[''])
# train_data=_file_to_word_ids('data/code/trainGT10.txt',voc)
# #
# # print(train_data)
# counter = collections.Counter(train_data)
# print('UNK:'+str(counter[9999]))
# print('all:'+str(sum(counter.values())))
# print(counter[9999]*1.0/sum(counter.values()))
# count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
# print(count_pairs)


# a=[1,3,7,9,1,4,7,9,5,4,6,8,9,3,2,4,8,2,6,8,5]
# print(a)
# b=[]
# index=0
# for i in range(len(a)):
#     if a[i]==4:
#         b.append(a[index:i+1])
#         index=i+1
# if index!=len(a) and len(a)-index>=4:
#     b.append(a[index:len(a)])
# print(b)
#
# num_step=4
# c=[]
# for i in range(len(b)):
#     if len(b[i])>num_step:
#         for index in range(len(b[i])-num_step+1):
#             c.append(b[i][index:index+num_step])
#     else:
#         c.append(b[i])
# print(c)

#获取转换为id的数据（train,valid,test)
def raw_data(data_path=None,voc_size=None):
    train_path=os.path.join(data_path,"trainGT10.txt")
    valid_path=os.path.join(data_path,"validGT10.txt")
    test_path=os.path.join(data_path,"testGT10.txt")

    word_to_id=_build_vocab(train_path,voc_size)
    train_data=_file_to_word_ids(train_path,word_to_id)
    valid_data=_file_to_word_ids(valid_path,word_to_id)
    test_data=_file_to_word_ids(test_path,word_to_id)

    vocabulary_size=len(word_to_id)
    end_id=word_to_id['ENDMARKER']
    return train_data,valid_data,test_data,word_to_id,vocabulary_size,end_id

def get_word_to_id(data_path=None,voc_size=None):
    train_path=os.path.join(data_path,"trainGT10.txt")
    word_to_id=_build_vocab(train_path,voc_size)
    return word_to_id

#将数据转换为n*numstep的形式，line[n+1]是line[n]左移一位（每个代码段末尾一行除外）
def split_data(train_data,ENDMARKER_id,num_step):
    _split_train_data=[]
    index=0
    for i in range(len(train_data)):
        if train_data[i]==ENDMARKER_id:
            _split_train_data.append(train_data[index:i+1])
            index=i+1
    if index!=len(train_data) and len(train_data)-index>=num_step:
        _split_train_data.append(train_data[index:len(train_data)])

    new_split_data=[]
    for i in range(len(_split_train_data)):
        if len(_split_train_data[i])>num_step:
            for index in range(len(_split_train_data[i])-num_step+1):
                new_split_data.append(_split_train_data[i][index:index+num_step])
        else:
            new_split_data.append(_split_train_data[i])
    return new_split_data


#TEST 将n*numstep形式的数据保存
# train_data,valid_data,test_data,vocabulary,vocab_size,end_id=raw_data('data/code/')
# print(end_id)
# _split_train_data=split_data(train_data,end_id,10)
# _split_test_data=split_data(test_data,end_id,10)
# _split_valid_data=split_data(valid_data,end_id,10)

#
# f=open('data/code/split_train.txt','w')
# for i in range(len(_split_train_data)):
#     f.write(str(_split_train_data[i])+'\n')
# f.close()
# f=open('data/code/split_test.txt','w')
# for i in range(len(_split_test_data)):
#     f.write(str(_split_test_data[i])+'\n')
# f.close()
# f=open('data/code/split_valid.txt','w')
# for i in range(len(_split_valid_data)):
#     f.write(str(_split_valid_data[i])+'\n')
# f.close()

#取出batchsize*numsteps大小的数据 index可以用queue
# return x y epoch_size

#消去UNK多的batch
def data_producer(data,batchsize,numsteps,word_to_id):
    x = []
    y = []
    X = []
    Y = []
    count = 0
    for i in range(len(data)):
        countUNK=0
        for j in range(numsteps):
            if data[i][j]==word_to_id['UNK']:
                countUNK+=1
        if countUNK*1.0/numsteps>=0.5:
            continue
        if data[i][numsteps-1]==word_to_id['ENDMARKER']:
            continue
        else:
            x.append(data[i])
            y.append(data[i+1])
            count+=1
        if count==batchsize:
            # print(batchsize)
            count=0
            X.append(x)
            Y.append(y)
            x=[]
            y=[]
    return X,Y,len(X)


#todo 消去UNK多的batch 后面解决变量名问题之后需要进行调整
def Data_producer(data,batch_size,numsteps,word_to_id):
    x = []
    y = []
    for i in range(len(data)):
        countUNK=0
        for j in range(numsteps):
            if data[i][j]==word_to_id['UNK']:
                countUNK+=1
        # 设置比例
        if countUNK*1.0/numsteps>=0.5:
            continue
        if data[i][numsteps-1]==word_to_id['ENDMARKER']:
            continue
        else:
            x.append(data[i])
            y.append(data[i+1])
    epoch_size=len(x)//batch_size
    return x,y,epoch_size

def stackHelper(x,numsteps,startid,endid):
    Helper = [[0 for col in range(numsteps)] for row in range(len(x))]
    for i in range(len(x)):
        for j in range(numsteps):
            if x[i][j]==startid:
                Helper[i][j]='START'
            elif x[i][j]==endid:
                Helper[i][j]='END'
    return Helper




#return a batch
def batch_producer(X,Y,index):
    print("index: "+str(index))
    return X[index],Y[index]

def Batch_producer(X,Y,batchsize,num_steps,epoch_size):
    X = tf.convert_to_tensor(X, name="X", dtype=tf.int32)
    Y = tf.convert_to_tensor(Y, name="Y", dtype=tf.int32)
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(X, [i*batchsize,0], [batchsize, num_steps])
    y = tf.slice(Y, [i*batchsize,0], [batchsize, num_steps])
    return x,y

#TEST
# X,Y,epochsize=data_producer(_split_train_data,20,10,end_id)
# x,y=producer(X,Y,0,10)
# print(X)
# print(Y)
# print(epochsize)
# q = queue.Queue(maxsize=epochsize)
# for i in range(epochsize):
#     q.put(i)
# for i in range(5):
#     print(batch_producer(X,Y,q.get()))







