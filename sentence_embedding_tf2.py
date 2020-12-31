#!/usr/bin/env python
#!-*- coding:utf-8 -*-

import jieba
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing

class DenseEmbeddingTag:
    def __init__(self, user_name):
        self.user_name = user_name

    def load_jiedai_data(self, file_name):
        wordcnt_dict = {}
        black_num = 0
        white_num = 0
        with open(file_name) as fp:
            lines = fp.readlines()
            for line in lines:
                label,desc=line.split("@@@@@@@@@@")[0],line.split("@@@@@@@@@@")[1]
                seg_list = self.cut_word(desc)
                wordcnt_dict = self.generate_wordcnt_dict(wordcnt_dict, seg_list)
                if int(label) == 1:
                    black_num += 1
                elif int(label) == 0:
                    white_num += 1
                #print('wordcnt_dict len: ', len(wordcnt_dict))
            fp.close()
        return black_num,white_num,wordcnt_dict

    #利用结巴库进行中文分词
    def cut_word(self, line):
        seg_list = jieba.cut(line, cut_all=True, HMM=True)
        return seg_list

    #利用jieba对每一行句子进行分词，接着统计每个分词的频次，存入wordcnt_dict字典中
    #目的是后续按照排序之后，对每个次进行编码（排序从4开始，如4~10000），编码之后对句子进行embedding
    def generate_wordcnt_dict(self, wordcnt_dict, seg_list):
        for seg in seg_list:
            if len(seg)>=1 and seg != '\n':
                if not seg in wordcnt_dict.keys():
                    wordcnt_dict[seg] = 1
                else:
                    wordcnt_dict[seg] += 1
        return wordcnt_dict
 
 
     #按照词出现的频次对词进行编码，比如出现最多的编码为4，第二多的编码为5，依次往后叠加
     def encode_word(self, wordcnt_dict):
        word_index_dict = {}
        wordcnt_list = sorted(wordcnt_dict.items(),key = lambda x:x[1], reverse=True)
        idx = 0
        word_index = 4
        for item in wordcnt_list:
            word_index_dict[item[0]] = word_index
            #if idx <= 100:
            #    print('word: ', item[0], 'word_cnt: ', item[1], 'word_index: ', word_index)
            word_index += 1
            idx += 1
        return word_index_dict

    #对排序后的字典进行encode编码，只取前面的word_num个词，不在字典中的，编码为2，在字典中，但是在word_num排序之后的，编码为3，其他的编码为对应的在字典中的序号
    #对每个句子都规范化为max_len个长度的list，比如256个，加上句子不满256个词，用0来填充，利用preprocessing.sequence.pad_sequences来实现
    def encode_train_data(self, file_name, sample_num, word_index_dict, word_num, max_len):
        lenp = len(range(0,sample_num))
        train_data = [0]*lenp
        train_labels = [0]*lenp
        train_sequences = [0]*lenp
        idx = 0
        with open(file_name) as fp:
            lines = fp.readlines()
            for line in lines:
                label,desc=line.split("@@@@@@@@@@")[0],line.split("@@@@@@@@@@")[1]
                train_labels[idx] = int(label)
                data = []
                seq_list = self.cut_word(desc)
                for seq in seq_list:
                    if not seq in word_index_dict.keys():
                        data.append(2)
                    else:
                        if word_index_dict[seq] < word_num:
                            data.append(word_index_dict[seq])
                        else:
                            data.append(3)
                train_data[idx] = data
                idx += 1
            fp.close()
        train_sequences = preprocessing.sequence.pad_sequences(train_data, max_len)
        
        #这里train_labels的shape是（None，1），不需要转换为onehot类型
        return ([train_data,train_labels, train_sequences])
 
    #加载需要预测的数据，
    def load_need_pred_data(self, file_name, word_index_dict, word_num, max_len):
        lenp = len(range(0,100000))
        need_pred_data = [0]*lenp
        need_pred_sequences = [0]*lenp
        need_pred_apk = [0]*lenp
        need_pred_desc = {}
        idx = 0
        with open(file_name) as fp:
            lines = fp.readlines()
            for line in lines:
                if len(line.split("@@@@@@@@@@")) != 2:
                    print('lines: ', lines)
                else:
                    apk,desc = line.split("@@@@@@@@@@")[0], line.split("@@@@@@@@@@")[1]
                    #print('apk: ', apk, 'desc: ', desc)
                    need_pred_desc[apk] = desc
                    need_pred_apk[idx] = apk
                    data = []
                    seq_list = self.cut_word(desc)
                    for seq in seq_list:
                        if not seq in word_index_dict.keys():
                            data.append(2)
                        else:
                            if word_index_dict[seq] < word_num:
                                data.append(word_index_dict[seq])
                            else:
                                data.append(3)
                    #print('idx:', idx, 'data: \n', data)
                    need_pred_data[idx] = data
                    idx += 1
            fp.close()
        #print('need_pred_data_len:\n', len(need_pred_data))
        #print('need_pred_data[0]:\n', need_pred_data[0])
        #print('need_pred_data[99]:\n', need_pred_data[99])

        need_pred_apk = need_pred_apk[0:idx]
        need_pred_sequences = preprocessing.sequence.pad_sequences(need_pred_data[0:idx], max_len)
        print('pred_data len: ', len(need_pred_sequences))
        return([need_pred_apk, need_pred_desc, need_pred_sequences])

    #构建CNN模型，第一层就是embedding层，实际上就是一个全连接层，把一个256长度的句子embedding为一个embedding_dim长度（本文用的是100）的数组，接着再进行CNN网络训练
    def model(self, train_sequences, train_labels, word_num, embedding_dim):
        model = tf.keras.Sequential()
        model.add(layers.Embedding(word_num, embedding_dim))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(128, activation=tf.nn.relu))
        model.add(layers.Dense(2, activation='softmax'))                        #注意，这里是softmax层，这种情况predict出来的代表两列代表0和1出现的概率，都在[0,1]的范围
        #model.add(layers.Dense(2))                                             #激活函数可以不用选softmax , 这种情况predict的数据格式预测出来例如：1.297   -0.175
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_sequences, train_labels, batch_size = 512, epochs = 10)

        return model

    def predict_new(self, model, need_pred_sequences):
        pred_result = model.predict(need_pred_sequences)
        #print('predict_result: ', pred_result, pred_result.shape)
        print('predict_result.shape: ', pred_result.shape)
        return(pred_result)


    def save_predict_result(self, file_name, need_pred_apk, need_pred_desc, predict_result):
        with open(file_name, "w") as fp:
            for idx in range(0,len(need_pred_apk)):
                apk = need_pred_apk[idx]
                if  apk in need_pred_desc.keys():
                    desc = need_pred_desc[apk]
                    white_pred_score = predict_result[idx][0]
                    black_pred_score = predict_result[idx][1]
                    fp.write("%.3f\t%s\t%s" % (black_pred_score, apk, desc))
            fp.close()
 
 if __name__ == '__main__':
    app_name_tag = DenseEmbeddingTag('app')
    print('load train_data file')
    black_num,white_num,wordcnt_dict = app_name_tag.load_jiedai_data("train_data.txt")
    print("black_num: ", black_num, "white_num: ", white_num, "word_cnt: ", len(wordcnt_dict))
    word_index_dict = app_name_tag.encode_word(wordcnt_dict)
    word_num = 10000
    embedding_dim = 100
    max_len = 256
    sample_num = black_num + white_num
    train_data,train_labels,train_sequences = app_name_tag.encode_train_data("train_data.txt", sample_num, word_index_dict, word_num, max_len)
    app_name_tag.print_data(train_data, train_labels, train_sequences)
    model = app_name_tag.model(train_sequences, train_labels, word_num, embedding_dim)
    need_pred_apk,need_pred_desc,need_pred_sequences = app_name_tag.load_need_pred_data("need_pred_data.txt", word_index_dict, word_num, max_len)
    #app_name_tag.predict("predict_result.txt", model, need_pred_apk, need_pred_sequences)
    predict_result = app_name_tag.predict_new(model, need_pred_sequences)
    app_name_tag.save_predict_result("predict_result.txt", need_pred_apk, need_pred_desc, predict_result)
 
