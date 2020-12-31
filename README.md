# tf2.0
tf2.0深度学习案例


train_data.csv 是一个文本训练的案例，按照@@@@@@@@@@来划分，第一列标签是0或者1，第二列是训练用到的文本(比如影评、情感分类的训练样本)

3、sentence_embedding_tf.py执行情况：

black_num:  686 white_num:  137415 word_cnt:  248507

train len:  138101 138101 138101

train_data:

 132 [3, 1522, 9, 32, 3258, 1228, 716, 27, 2, 2, 484, 5336, 1146, 2117, 78, 7, 841, 97, 1634, 1494, 167, 2, 2, 9, 983, 1805, 3, 182, 206, 155, 6245, 18, 2, 2, 14, 25, 20, 4783, 4417, 114, 2, 2, 305, 2, 2, 2300, 3, 716, 1259, 18, 2, 2, 2998, 422, 2, 2, 2059, 282, 2, 2, 7542, 916, 2, 2, 88, 151, 716, 786, 2, 2, 6, 5, 2, 2, 6, 493, 2, 2, 2, 140, 2, 2, 401, 131, 3885, 232, 2, 2, 441, 33, 1004, 2, 2, 1259, 2, 2, 786, 548, 21, 2, 2, 2359, 2059, 2331, 88, 2, 2, 1228, 2, 2, 1259, 548, 21, 2, 2, 664, 77, 2465, 2, 2, 21, 2, 2, 23, 258, 42, 2, 2, 16, 33, 731, 2]
 
train_sequences:

 256 [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    3 1522
    9   32 3258 1228  716   27    2    2  484 5336 1146 2117   78    7
  841   97 1634 1494  167    2    2    9  983 1805    3  182  206  155
 6245   18    2    2   14   25   20 4783 4417  114    2    2  305    2
    2 2300    3  716 1259   18    2    2 2998  422    2    2 2059  282
    2    2 7542  916    2    2   88  151  716  786    2    2    6    5
    2    2    6  493    2    2    2  140    2    2  401  131 3885  232
    2    2  441   33 1004    2    2 1259    2    2  786  548   21    2
    2 2359 2059 2331   88    2    2 1228    2    2 1259  548   21    2
    2  664   77 2465    2    2   21    2    2   23  258   42    2    2
   16   33  731    2]
   
train_labels:

 1
 
 Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 100)         1000000
_________________________________________________________________
global_average_pooling1d (Gl (None, 100)               0
_________________________________________________________________
dense (Dense)                (None, 128)               12928
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 258
=================================================================
Total params: 1,013,186
Trainable params: 1,013,186
Non-trainable params: 0

Epoch 1/10
138101/138101 [==============================] - 10s 72us/sample - loss: 0.0739 - accuracy: 0.9917

Epoch 2/10
138101/138101 [==============================] - 10s 70us/sample - loss: 0.0073 - accuracy: 0.9978

Epoch 3/10
138101/138101 [==============================] - 10s 70us/sample - loss: 0.0052 - accuracy: 0.9985

Epoch 4/10
138101/138101 [==============================] - 10s 71us/sample - loss: 0.0043 - accuracy: 0.9988

Epoch 5/10
138101/138101 [==============================] - 10s 72us/sample - loss: 0.0036 - accuracy: 0.9990

Epoch 6/10
138101/138101 [==============================] - 10s 70us/sample - loss: 0.0030 - accuracy: 0.9992

Epoch 7/10
138101/138101 [==============================] - 10s 71us/sample - loss: 0.0025 - accuracy: 0.9993

Epoch 8/10
138101/138101 [==============================] - 10s 71us/sample - loss: 0.0020 - accuracy: 0.9994

Epoch 9/10
138101/138101 [==============================] - 10s 72us/sample - loss: 0.0017 - accuracy: 0.9995

Epoch 10/10
138101/138101 [==============================] - 10s 73us/sample - loss: 0.0015 - accuracy: 0.9995

pred_data len:  87525
predict_result.shape:  (87525, 2)



3、sentence_lstm_embedding_tf.py

与上一个模型相比，唯一区别的就是model的构造有所区别

def model(self, train_sequences, train_labels, word_num, embedding_dim):

        model = tf.keras.Sequential()
        model.add(layers.Embedding(word_num, embedding_dim))
        #model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Bidirectional(layers.LSTM(64)))
        model.add(layers.Dense(128, activation=tf.nn.relu))
        model.add(layers.Dense(2, activation='softmax'))
        #model.add(layers.Dense(1))
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        model.fit(train_sequences, train_labels, batch_size = 512, epochs = 3)

执行输出情况：
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 100)         1000000
_________________________________________________________________
bidirectional (Bidirectional (None, 128)               84480
_________________________________________________________________
dense (Dense)                (None, 128)               16512
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129
=================================================================
Total params: 1,101,121
Trainable params: 1,101,121
Non-trainable params: 0
_________________________________________________________________
None

Epoch 1/5

138101/138101 [==============================] - 257s 2ms/sample - loss: 0.1938 - accuracy: 0.9950

Epoch 2/5

138101/138101 [==============================] - 251s 2ms/sample - loss: 0.0226 - accuracy: 0.9950

Epoch 3/5

138101/138101 [==============================] - 252s 2ms/sample - loss: 0.0195 - accuracy: 0.9951

Epoch 4/5

138101/138101 [==============================] - 252s 2ms/sample - loss: 0.0189 - accuracy: 0.9964

Epoch 5/5
138101/138101 [==============================] - 253s 2ms/sample - loss: 0.0115 - accuracy: 0.9972
 

6、textCnn_embedding_tf2.py执行情况

原理可参考：https://zhuanlan.zhihu.com/p/96044890

Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 256)]        0
__________________________________________________________________________________________________
embedding (Embedding)           (None, 256, 100)     1000000     input_1[0][0]

__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 254, 128)     38528       embedding[0][0]

__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 253, 128)     51328       embedding[0][0]

__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 252, 128)     64128       embedding[0][0]

__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 128)          0           conv1d[0][0]

__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]

__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]

__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 384)          0           global_max_pooling1d[0][0]

                                                                 global_max_pooling1d_1[0][0]
                                                                 
                                                                 global_max_pooling1d_2[0][0]
                                                                 
__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            770         concatenate[0][0]

==================================================================================================
Total params: 1,154,754

Trainable params: 1,154,754

Non-trainable params: 0

__________________________________________________________________________________________________
None

Epoch 1/5

138101/138101 [==============================] - 104s 755us/sample - loss: 0.0276 - accuracy: 0.9963

Epoch 2/5

138101/138101 [==============================] - 104s 752us/sample - loss: 0.0051 - accuracy: 0.9983

Epoch 3/5

138101/138101 [==============================] - 103s 749us/sample - loss: 0.0032 - accuracy: 0.9989

Epoch 4/5

138101/138101 [==============================] - 104s 750us/sample - loss: 0.0019 - accuracy: 0.9994

Epoch 5/5

138101/138101 [==============================] - 104s 752us/sample - loss: 0.0011 - accuracy: 0.9997

pred_data len:  87525

predict_result.shape:  (87525, 2)



