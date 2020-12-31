#!/usr/bin/env python
#!-*- coding:utf-8 -*-

import jieba
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model

class DenseEmbeddingTag:
    def __init__(self, user_name):
        self.user_name = user_name

    def model(self, train_sequences, train_labels, word_num, embedding_dim):
        model = tf.keras.Sequential()
        model.add(layers.Embedding(word_num, embedding_dim))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(128, activation=tf.nn.relu))
        model.add(layers.Dense(2, activation='softmax'))
        #model.add(layers.Dense(1))
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_sequences, train_labels, batch_size = 512, epochs = 5)

        return model

    def middle_layer(self, model, need_pred_sequences):
        embedding_layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
        print(embedding_layer_model.summary())
        for layer in model.layers:
            weights = layer.get_weights() 
            for idx in range(len(weights)):
                print('weights: ', idx," shape: ", weights[idx].shape,"\n", weights[idx])
        #embedding_layer_pred_result = embedding_layer_model.predict(need_pred_sequences)
        #print('embedding_layer_pred_result.shape', embedding_layer_pred_result.shape)

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
    app_name_tag.middle_layer(model, need_pred_sequences)
