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


8、textRnn_embedding.py执行情况
Model: "model"

_________________________________________________________________
Layer (type)                 Output Shape              Param #

=================================================================
input_1 (InputLayer)         [(None, 256)]             0

_________________________________________________________________
embedding (Embedding)        (None, 256, 100)          1000000

_________________________________________________________________
bidirectional (Bidirectional (None, 256)               234496

_________________________________________________________________
dense (Dense)                (None, 2)                 514

=================================================================
Total params: 1,235,010

Trainable params: 1,235,010

Non-trainable params: 0

_________________________________________________________________
None
Epoch 1/5

138101/138101 [==============================] - 406s 3ms/sample - loss: 0.0371 - accuracy: 0.9957

Epoch 2/5

138101/138101 [==============================] - 400s 3ms/sample - loss: 0.0287 - accuracy: 0.9933

Epoch 3/5

138101/138101 [==============================] - 403s 3ms/sample - loss: 0.0244 - accuracy: 0.9952

Epoch 4/5

138101/138101 [==============================] - 399s 3ms/sample - loss: 0.0094 - accuracy: 0.9969

Epoch 5/5

138101/138101 [==============================] - 403s 3ms/sample - loss: 0.0065 - accuracy: 0.9981

pred_data len:  87525

predict_result.shape:  (87525, 2)



9、textRCnn_embedding.py

与RNN以及CNN不同的地方在于，有三个input，同时进行fit训练拟合是，输入的train_x是train_sequences_new = [train_sequences, train_sequences, train_sequences]是三个相同的输入

def text_rcnn_model(self, train_sequences, train_labels, word_num, embedding_dim, max_len):
        input_current = Input((max_len,))
        input_left = Input((max_len,))
        input_right = Input((max_len,))
        embedding_current = layers.Embedding(word_num, embedding_dim, input_length=max_line_len)(input_current)
        embedding_left = layers.Embedding(word_num, embedding_dim, input_length=max_line_len)(input_left)
        embedding_right = layers.Embedding(word_num, embedding_dim, input_length=max_line_len)(input_right)

        x_left = layers.SimpleRNN(128, return_sequences=True)(embedding_left)
        x_right = layers.SimpleRNN(128, return_sequences=True,go_backwards=True)(embedding_right)
        x_right = layers.Lambda(lambda x: keras.backend.reverse(x, axes=1))(x_right)
        x = layers.Concatenate(axis=2)([x_left, embedding_current, x_right])
        x = layers.GlobalMaxPooling1D()(x)

        output = layers.Dense(2, activation='softmax')(x)
        model = Model(inputs = [input_current, input_left, input_right], outputs = output)
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
        train_sequences_new = [train_sequences, train_sequences, train_sequences]
        
        model.fit(train_sequences_new, train_labels, batch_size = 512, epochs = 5)
        return(model)

预测predict时也是类似的三个输入：
def predict_new(self, model, need_pred_sequences):
        need_pred_sequences = [need_pred_sequences, need_pred_sequences, need_pred_sequences]
        pred_result = model.predict(need_pred_sequences)
        #print('predict_result: ', pred_result, pred_result.shape)
        print('predict_result.shape: ', pred_result.shape)
        return(pred_result)

执行输出情况：
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to

==================================================================================================
input_3 (InputLayer)            [(None, 256)]        0

__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 256)]        0

__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 256, 100)     1000000     input_3[0][0]

__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 256, 100)     1000000     input_2[0][0]

__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 256)]        0

__________________________________________________________________________________________________
simple_rnn_1 (SimpleRNN)        (None, 256, 128)     29312       embedding_2[0][0]

__________________________________________________________________________________________________
simple_rnn (SimpleRNN)          (None, 256, 128)     29312       embedding_1[0][0]

__________________________________________________________________________________________________
embedding (Embedding)           (None, 256, 100)     1000000     input_1[0][0]

__________________________________________________________________________________________________
lambda (Lambda)                 (None, 256, 128)     0           simple_rnn_1[0][0]

__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 256, 356)     0         
simple_rnn[0][0]
                                                                 embedding[0][0]
                                                                 
                                                                 lambda[0][0]
                                                                 
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 356)          0           concatenate[0][0]

__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            714         global_max_pooling1d[0][0]

==================================================================================================
Total params: 3,059,338

Trainable params: 3,059,338

Non-trainable params: 0

__________________________________________________________________________________________________
None
Epoch 1/5

138101/138101 [==============================] - 200s 1ms/sample - loss: 0.0309 - accuracy: 0.9940

Epoch 2/5

138101/138101 [==============================] - 196s 1ms/sample - loss: 0.0078 - accuracy: 0.9975

Epoch 3/5

114176/138101 [=======================>......] - ETA: 33s - loss: 0.0046 - accuracy: 0.9985


9，textSelfAttention的执行情况

WQ.shape (None, 256, 100)
K.permute_dimensions(WK, [0, 2, 1]).shape (None, 100, 256)
QK.shape (None, 256, 256)
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 256)]             0
_________________________________________________________________
embedding (Embedding)        (None, 256, 100)          1000000
_________________________________________________________________
self_attention (SelfAttentio (None, 256, 100)          30000
_________________________________________________________________
global_average_pooling1d (Gl (None, 100)               0
_________________________________________________________________
dropout (Dropout)            (None, 100)               0
_________________________________________________________________
dense (Dense)                (None, 2)                 202
=================================================================
Total params: 1,030,202
Trainable params: 1,030,202
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/5
138101/138101 [==============================] - 162s 1ms/sample - loss: 0.0601 - accuracy: 0.9959
Epoch 2/5
138101/138101 [==============================] - 160s 1ms/sample - loss: 0.0062 - accuracy: 0.9982
Epoch 3/5
138101/138101 [==============================] - 159s 1ms/sample - loss: 0.0055 - accuracy: 0.9983
Epoch 4/5
138101/138101 [==============================] - 157s 1ms/sample - loss: 0.0046 - accuracy: 0.9985
Epoch 5/5
138101/138101 [==============================] - 158s 1ms/sample - loss: 0.0041 - accuracy: 0.9987

10、textBiLSTM_Attention_tf2.py的网络情况

attention __init__, step_dim:  256
attention build input_shape:
 3 (None, 256, 256)
input_shape[-1]:  256
attention call x.shape:  (None, 256, 256) features_dim:  256 step_dim:  256
self.W.shape:  (256,) keras.backend.reshape(self.W, (features_dim, 1)).shape:  (256, 1)
keras.backend.reshape(x,(-1,features_dim).shape):  (None, 256)
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 256)]             0
_________________________________________________________________
embedding (Embedding)        (None, 256, 100)          1000000
_________________________________________________________________
bidirectional (Bidirectional (None, 256, 256)          234496
_________________________________________________________________
attention (Attention)        (None, 256)               256
_________________________________________________________________
dense (Dense)                (None, 2)                 514
=================================================================
Total params: 1,235,266
Trainable params: 1,235,266
Non-trainable params: 0
_________________________________________________________________
None


保存以及加载模型时，包含有自己定义的layer时，会出现一些问题，解决方法可参考：
https://zhuanlan.zhihu.com/p/86886620

另外需要把模型保存为特定pb格式时，使用接口：tf.saved_model.save(model, 'PB_MODEL_FILE') #后面参数是文件夹
方法可参考：https://zhuanlan.zhihu.com/p/146243327



