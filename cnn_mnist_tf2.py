import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train,x_test = x_train/255.0, x_test/255.0

print('train.shape: ', x_train.shape, y_train.shape)   # (60000, 28, 28), (60000,)
print('test.shape: ', x_test.shape, y_test.shape)    # (10000, 28, 28), (10000,)


model = tf.keras.Sequential()
model.add(tf.keras.layer.Reshape((28,28,1), input_shape=(28,28)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))

model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))   # 最后一层要换成softmax

model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

onehot_y_train = tf.keras.utils.to_categorical(y_train)  # 因为最后一层是softmax，要转换为onehot类型(60000, 10)
onehot_y_test = tf.keras.utils.to_categorical(y_test)

model.fit(x_train, onehot_y_train, epochs = 10, batch_size = 1000)

model.evaluate(x_test, onehot_y_test, verbose=2)
