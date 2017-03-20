# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import collections
#读取文件，将token以list形式保存
def _read_words(filename):
    with tf.gfile.GFile(filename,'r') as f:
        return f.read().decode('utf-8').replace("\r\n"," ENDMARKER ").split(' ')
        # return f.read().decode('utf-8').split(' ')
#
# data=_read_words('data/code/trainLineGT10.txt')
# f=open('data/code/tmp.txt','w')
# f.write(str(data))

#创建词汇表，将token按出现频率排序，转换为word:id 大小为94
def _build_vocab(filename):
    data = _read_words(filename)
    # print(data)
    counter = collections.Counter(data)
    # print(counter[''])
    # print(sum(counter.values()))
    # a=sum(counter.values())
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, values = list(zip(*count_pairs))

    word_to_id = dict(zip(words, range(len(words))))
    word_to_id['ENDTOK']=len(word_to_id)
    return word_to_id

# vocab=_build_vocab('D:/py_project/Tensorflow/s-lstm/data/train.txt')
# vocab = sorted(vocab.items(), key=lambda d: d[1])
# print(vocab)

# data = _read_words('D:/py_project/Tensorflow/s-lstm/data/test.txt')
# counter = collections.Counter(data)
# # print(counter)
# print(len(counter))
# count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
# words, values = list(zip(*count_pairs))
# print(words)

#将文件中的token转换为id
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    wordId=[]
    for word in data:
        if word in word_to_id:
            wordId.append(word_to_id[word])
        # else:
        #     print(word)
        #     print('not in voc')
    # return [word_to_id[word] for word in data if word in word_to_id]
    return wordId

# #得到token id  10为endmarker
# voc=_build_vocab('D:/py_project/Tensorflow/s-lstm/data/test.txt')
# print(voc)
# # print(len(voc))
# print(voc[''])
# train_data=_file_to_word_ids('D:/py_project/Tensorflow/s-lstm/data/train.txt',voc)
# print(train_data)
# test_data=_file_to_word_ids('D:/py_project/Tensorflow/s-lstm/data/test.txt',voc)
# print(test_data)

#获取转换为id的数据（train,valid,test)
def raw_data(data_path=None,word_to_id=None):
    train_path=os.path.join(data_path,"new_train.txt")
    test_path=os.path.join(data_path,"new_test.txt")

    word_to_id=word_to_id
    train_data=_file_to_word_ids(train_path,word_to_id)
    test_data=_file_to_word_ids(test_path,word_to_id)

    vocabulary_size=len(word_to_id)
    end_id=word_to_id['ENDMARKER']
    left_id=word_to_id['{']
    right_id=word_to_id['}']
    END_ID=word_to_id['ENDTOK']
    return train_data,test_data,vocabulary_size,end_id,END_ID,left_id,right_id

def get_word_to_id(data_path=None):
    train_path=os.path.join(data_path,"test.txt")
    word_to_id=_build_vocab(train_path)
    return word_to_id

#X[i]为第i行代码  Y[i]为X[i]左移一位，最后一位用ENDTOK补全
def split_data(data,word_to_id):
    _split_train_data=[]
    index=0
    for i in range(len(data)):
        if data[i]==word_to_id["ENDMARKER"]:
            _split_train_data.append(data[index:i+1])
            index=i+1
    # if index!=len(train_data) and len(train_data)-index>=num_step:
    #     _split_train_data.append(train_data[index:len(train_data)])
    X=[]
    Y=[]
    for i in range(len(_split_train_data)):
        newdata=_split_train_data[i]
        newdata.append(word_to_id['ENDTOK'])
        X.append(newdata[:len(newdata)-1])
        Y.append(newdata[1:])
    epoch_size=len(X)
    return X,Y,epoch_size

#
# train_data,test_data,vocabulary,vocab_size,end_id,END_ID,left_id,right_id=raw_data('D:/py_project/Tensorflow/s-lstm/data/')
# print(vocab_size)
# X,Y,_=split_data(train_data,vocabulary)
# print(Y)

#取出batchsize*numsteps大小的数据 index可以用queue
# return x y epoch_size

# def Data_producer(data,batch_size,numsteps,word_to_id):
#     x = []
#     y = []
#     for i in range(len(data)):
#         if data[i][numsteps-1]==word_to_id['ENDMARKER']:
#             continue
#         else:
#             x.append(data[i])
#             y.append(data[i+1])
#     epoch_size=len(x)//batch_size
#     return x,y,epoch_size

def stackHelper(x,startid,endid):
    Helper = [[0 for col in range(len(x[row]))] for row in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j]==startid:
                Helper[i][j]='START'
            elif x[i][j]==endid:
                Helper[i][j]='END'
    return Helper

#todo fix me
def Batch_producer(X,Y,index):
    print("INDEX in batch producer: %d"%index)
    x=[X[index]]
    y=[Y[index]]
    x = tf.convert_to_tensor(x, name="x", dtype=tf.int32)
    y = tf.convert_to_tensor(y, name="y", dtype=tf.int32)
    length=len(X[index])
    print("NUMSTEPS: %d"%length)
    return x,y,length

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



#
f=open('D:/py_project/Tensorflow/s-lstm/data/test.txt')
f1=open('D:/py_project/Tensorflow/s-lstm/data/new_test.txt','w')
lineNUM=0
while 1:
    line=f.readline()
    if not line:
        break
    code=line.split(' ')
    if lineNUM<600:
        if len(code)<=600:
            f1.write(line)
            lineNUM+=1
    else: break


