import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0

print('train_shape: ', train_x.shape, train_y.shape)    #(50000, 32, 32, 3),  (50000, 1)
print('train_label[0]:',  train_y[0])                               #[6]，不是onehot类型


#构建多层CNN卷积神经网络
model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((32,32,3), input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())                                                          #全连接之前，需要进行flatten转换
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))                                     #最后一层是softmax层，2分类问题

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

onehot_train_y = tf.keras.utils.to_categorical(train_y)                   #由于最后一层是softmax层，需要转换为onehot类型，shape为（50000， 10）
onehot_test_y = tf.keras.utils.to_categorical(test_y)


model.fit(train_x, onehot_train_y, epochs=10, batch_size=5000, validation_data=(test_x, onehot_test_y))

model.evaluate(test_x, onehot_test_y)

