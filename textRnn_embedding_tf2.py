#!/usr/bin/env python
#!-*- coding:utf-8 -*-

import jieba
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input,Model
from tensorflow.keras import preprocessing

class TextRnnTag:
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
                
    
    def cut_word(self, line):
        seg_list = jieba.cut(line, cut_all=True, HMM=True)
        return seg_list

    def generate_wordcnt_dict(self, wordcnt_dict, seg_list):
        for seg in seg_list:
            if len(seg)>=1 and seg != '\n':
                if not seg in wordcnt_dict.keys():
                    wordcnt_dict[seg] = 1
                else:
                    wordcnt_dict[seg] += 1
        return wordcnt_dict

    def encode_word(self, wordcnt_dict):
        word_index_dict = {}
        wordcnt_list = sorted(wordcnt_dict.items(),key = lambda x:x[1], reverse=True)
        idx = 0
        word_index = 3
        for item in wordcnt_list:
            word_index_dict[item[0]] = word_index
            #if idx <= 100:
            #    print('word: ', item[0], 'word_cnt: ', item[1], 'word_index: ', word_index)
            word_index += 1    
            idx += 1
        return word_index_dict 

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
        return ([train_data,train_labels, train_sequences]) 


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
   
    def text_rnn_model(self, train_sequences, train_labels, word_num, embedding_dim, max_len):
        input = Input((max_len,))
        embedding = layers.Embedding(word_num, embedding_dim, input_length = max_len)(input)
        bi_lstm = layers.Bidirectional(layers.LSTM(128))(embedding)
        output = layers.Dense(2, activation='softmax')(bi_lstm)
        model = Model(inputs = input, outputs = output)
        print(model.summary())
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_sequences, train_labels, batch_size = 512, epochs = 5)

        #input = Input((max_len,))
        #embedding = layers.Embedding(word_num, embedding_dim, input_length=max_line_len)(input)
        #convs = []
        #for kernel_size in [ 3, 4, 5]:
        #    c = layers.Conv1D(128, kernel_size, activation='relu')(embedding)
        #    c = layers.GlobalMaxPooling1D()(c)
        #    convs.append(c) 
        #x = layers.Concatenate()(convs)
        #output = layers.Dense(2, activation='softmax')(x)
        #model = Model(inputs = input, outputs = output)
        #print(model.summary())

        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #model.fit(train_sequences, train_labels, batch_size = 512, epochs = 5)
        return(model)



    def model(self, train_sequences, train_labels, word_num, embedding_dim):
        model = tf.keras.Sequential()
        model.add(layers.Embedding(word_num, embedding_dim))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(128, activation=tf.nn.relu))
        model.add(layers.Dense(2, activation='softmax'))
        #model.add(layers.Dense(1))
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_sequences, train_labels, batch_size = 512, epochs = 10)

        return model

    def predict_new(self, model, need_pred_sequences):
        pred_result = model.predict(need_pred_sequences)
        #print('predict_result: ', pred_result, pred_result.shape)
        print('predict_result.shape: ', pred_result.shape)
        return(pred_result)


    def predict(self, file_name,  model, need_pred_apk, need_pred_sequences):
        idx = 0
        with open(file_name, "w") as fp:
            for sequence in need_pred_sequences:
                data = [0]*1
                data[0] = sequence
                pred_result = model.predict(data)
                if idx <= 2:
                    print('idx: ', idx,'apk: ', need_pred_apk[idx], 'sequences: ', len(data),sequence) 
                    print('predict_result: ', pred_result, pred_result.shape)
                idx += 1
           
            fp.close()

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


    def print_data(self, train_data, train_labels, train_sequences):
        print('train len: ', len(train_data),  len(train_labels), len(train_sequences))
        for idx in range(0,3):
            print('train_data: \n', len(train_data[idx]), train_data[idx])
            print('train_sequences: \n', len(train_sequences[idx]), train_sequences[idx])
            print('train_labels: \n', train_labels[idx])


if __name__ == '__main__':
    app_name_tag = TextRnnTag('text rnn model')
    print('load train_data file')
    black_num,white_num,wordcnt_dict = app_name_tag.load_jiedai_data("../train_data.txt")
    print("black_num: ", black_num, "white_num: ", white_num, "word_cnt: ", len(wordcnt_dict))
    word_index_dict = app_name_tag.encode_word(wordcnt_dict)
    word_num = 10000
    embedding_dim = 100
    max_len = 256
    max_line_len = 1000000
    sample_num = black_num + white_num
    train_data,train_labels,train_sequences = app_name_tag.encode_train_data("../train_data.txt", sample_num, word_index_dict, word_num, max_len)
    app_name_tag.print_data(train_data, train_labels, train_sequences)
    model = app_name_tag.text_rnn_model(train_sequences, train_labels,word_num, embedding_dim, max_len)
    #model = app_name_tag.model(train_sequences, train_labels, word_num, embedding_dim) 
    need_pred_apk,need_pred_desc,need_pred_sequences = app_name_tag.load_need_pred_data("../need_pred_data.txt", word_index_dict, word_num, max_len)
    predict_result = app_name_tag.predict_new(model, need_pred_sequences)
    app_name_tag.save_predict_result("predict_result.txt", need_pred_apk, need_pred_desc, predict_result)


