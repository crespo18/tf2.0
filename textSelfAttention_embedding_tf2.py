#!/usr/bin/env python
#!-*- coding:utf-8 -*-

import jieba
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input,Model
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Layer
#from keras.engine.topology import Layer                                        #tensorflow2.4以上版本可以用，2.0版本的用上面的tensorflow.keras.layers
from tensorflow.keras import initializers, regularizers, constraints
import pandas as pd
from keras.optimizers import SGD,Adam
from keras import backend as K

class SelfAttention(Layer):

    def __init__(self, output_dim, name="SelfAttention", **kwargs):
        self.output_dim = output_dim
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(SelfAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)


        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (self.output_dim**0.5)

        QK = K.softmax(QK)

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)

    #当需要save以及load_mode的模型中，保存有自己定义的layer时，务必需要实现get_config，把__init__里面的参数带到字典里面，不然会失败
    def get_config(self):
        config = {"output_dim":self.output_dim}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
    
    def text_self_attention_model(self, train_sequences, train_labels, word_num, embedding_dim, max_len, model_file, pb_model_file):
        S_inputs = Input(shape=(max_len,), dtype='int32')
        embedding = layers.Embedding(word_num, embedding_dim)(S_inputs)
        x = layers.LSTM(128,return_sequences=True)(embedding)
        O_seq = SelfAttention(embedding_dim)(x)
        O_seq = layers.GlobalAveragePooling1D()(O_seq)
        O_seq = layers.Dropout(0.5)(O_seq)
        outputs = layers.Dense(2, activation='softmax')(O_seq)
        model = Model(inputs=S_inputs, outputs=outputs)
        print(model.summary())

        #opt = Adam(lr=0.0002,decay=0.00001)                  #2.4版本可以用
        #loss = 'categorical_crossentropy' 

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        
        model.fit(train_sequences, train_labels, batch_size = 512, epochs = 5)
        
        #模型保存
        model.save(model_file)
        
        #如果需要利用TF Serving来部署模型的话，要保存为pb格式的model
        tf.saved_model.save(model, pb_model_file)
        return(model)

    
    def predict_with_model_file(self, model_file, need_pred_sequences):
        #当加载的模型包含着自己定义的layer时，务必需要传入custom_objects这个字典参数，带上自己定义的layer的关键字
        model = tf.keras.models.load_model(model_file, custom_objects={"SelfAttention":SelfAttention})
        pred_result = model.predict(need_pred_sequences)
        #print('predict_result: ', pred_result, pred_result.shape)
        print('predict_result.shape: ', pred_result.shape)
        return(pred_result)

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
    print(tf.__version__)
    app_name_tag = DenseEmbeddingTag('app')
    print('load train_data file')
    black_num,white_num,wordcnt_dict = app_name_tag.load_jiedai_data("../train_data.txt")
    print("black_num: ", black_num, "white_num: ", white_num, "word_cnt: ", len(wordcnt_dict))
    word_index_dict = app_name_tag.encode_word(wordcnt_dict)
    word_num = 10000
    embedding_dim = 100
    max_len = 256
    max_line_len = 1000000
    model_file = 'MODEL_FILE/lstm_self_attention.model'
    pb_model_file = 'PB_MODEL_FILE'
    sample_num = black_num + white_num
    train_data,train_labels,train_sequences = app_name_tag.encode_train_data("../train_data.txt", sample_num, word_index_dict, word_num, max_len)
    app_name_tag.print_data(train_data, train_labels, train_sequences)
    #train_labels = tf.keras.utils.to_categorical(train_labels)
    #train_labels = pd.get_dummies(train_labels)                                                                      #转换为onehot类型，看情况用
    model = app_name_tag.text_self_attention_model(train_sequences, train_labels,word_num, embedding_dim, max_len, model_file, pb_model_file)
    need_pred_apk,need_pred_desc,need_pred_sequences = app_name_tag.load_need_pred_data("../need_pred_data.txt", word_index_dict, word_num, max_len)
    #predict_result = app_name_tag.predict_with_model_file(model_file, need_pred_sequences)
    predict_result = app_name_tag.predict_new(model, need_pred_sequences)
    app_name_tag.save_predict_result("predict_result.txt", need_pred_apk, need_pred_desc, predict_result)


