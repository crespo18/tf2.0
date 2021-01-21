#!/usr/bin/env python
#!-*- coding:utf-8 -*-

import jieba
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input,Model
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
import pandas as pd


class Attention(Layer):
    def __init__(self, step_dim, name="Attention", W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=None, **kwargs):
        print('attention __init__, step_dim: ', step_dim)
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
       
        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer
       
        self.W_constraint = W_constraint
        self.b_constraint = b_constraint
        
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
  
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        print('attention build input_shape:\n', len(input_shape), input_shape)
        print('input_shape[-1]: ', input_shape[-1])
        #assert len(input_shape) == 3 

        self.W = self.add_weight(shape=(input_shape[-1], ),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)   
        self.features_dim = input_shape[-1]
        
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None 

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        print('attention call x.shape: ', x.shape, 'features_dim: ', features_dim, 'step_dim: ', step_dim)

        print('self.W.shape: ', self.W.shape, 'keras.backend.reshape(self.W, (features_dim, 1)).shape: ', keras.backend.reshape(self.W, (features_dim, 1)).shape)
        print('keras.backend.reshape(x,(-1,features_dim).shape): ', keras.backend.reshape(x, (-1, features_dim)).shape)

        e = keras.backend.reshape(keras.backend.dot(keras.backend.reshape(x, (-1, features_dim)), keras.backend.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = keras.backend.tanh(e)

        a = keras.backend.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= keras.backend.cast(mask, keras.backend.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= keras.backend.cast(keras.backend.sum(a, axis=1, keepdims=True) + keras.backend.epsilon(), keras.backend.floatx())
        a = keras.backend.expand_dims(a)

        c = keras.backend.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        print('attention compute_output_shape: input_shape', input_shape, 'feature_dim: ', self.features_dim)
        return (input_shape[0], self.features_dim)

    #如果需要保存的模型中，含有自己自定义的layer，必须要自己重写get_config函数，把需要的变量增加到字典中，这里增加的是__init__函数里面的第一个非默认的参数step_dim
    #如果不重写get_config，在save或者load_model的时候会出错
    def get_config(self):
        config = {"step_dim":self.step_dim}
        base_config = super(Attention, self).get_config()
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
   
    def text_bi_lstm_attention_model(self, train_sequences, train_labels, word_num, embedding_dim, max_len, model_file):
        input = Input((max_len,))
        embedding = layers.Embedding(word_num, embedding_dim, input_length=max_line_len)(input)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(embedding)      #这里用的是双向LSTM，也可以直接用单向LSTM，layer.LSTM()，
                                                                                          #注意attention要求输入的input_shpe是3个维度的，如(None, 256, 100)，所以上一层的LSTM需要加上
                                                                                          #return_sequences=True，如果不加的话，LSTM只是输出最后一个状态，那就是2个维度（None，128）
                                                                                          #加上之后，输出的维度是（None，256,128），因为加上之后保存了每个词的隐形向量，不加的话仅仅是
                                                                                          #保持最后一个词的隐形向量
        #x = layers.LSTM(128, return_sequences=True)(embedding)
        x = Attention(max_len)(x)
        output = layers.Dense(2, activation='softmax')(x)
        model = Model(inputs = input, outputs = output)
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_sequences, train_labels, batch_size = 512, epochs = 5)
        
        #模型保存，这里务必要注意上面自定义的attention这个layer，必须要重写get_config函数
        model.save(model_file)
        return(model)
    
    def predict_with_model_file(self, model_file, need_pred_sequences):
        #由于模型里面包含着自己定义的attention这个layer，在load_model的时候必须要增加custom_objuect这个字典参数，传入Attention关键字
        model = tf.keras.models.load_model(model_file, custom_objects={"Attention":Attention})
        pred_result = model.predict(need_pred_sequences)
        #print('predict_result: ', pred_result, pred_result.shape)
        print('predict_result.shape: ', pred_result.shape)
        return(pred_result)

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
    black_num,white_num,wordcnt_dict = app_name_tag.load_jiedai_data("../train_data.txt")
    print("black_num: ", black_num, "white_num: ", white_num, "word_cnt: ", len(wordcnt_dict))
    word_index_dict = app_name_tag.encode_word(wordcnt_dict)
    word_num = 10000
    embedding_dim = 100
    max_len = 256
    max_line_len = 1000000
    model_file = 'MODEL_FILE/bilstm_attention.model'
    sample_num = black_num + white_num
    train_data,train_labels,train_sequences = app_name_tag.encode_train_data("../train_data.txt", sample_num, word_index_dict, word_num, max_len)
    app_name_tag.print_data(train_data, train_labels, train_sequences)
    #train_labels = tf.keras.utils.to_categorical(train_labels)
    #train_labels = pd.get_dummies(train_labels)
    model = app_name_tag.text_bi_lstm_attention_model(train_sequences, train_labels,word_num, embedding_dim, max_len, model_file)
    need_pred_apk,need_pred_desc,need_pred_sequences = app_name_tag.load_need_pred_data("../need_pred_data.txt", word_index_dict, word_num, max_len)
    predict_result = app_name_tag.predict_new(model, need_pred_sequences)
    #predict_result = app_name_tag.predict_with_model_file(model_file, need_pred_sequences)
    app_name_tag.save_predict_result("predict_result.txt", need_pred_apk, need_pred_desc, predict_result)


