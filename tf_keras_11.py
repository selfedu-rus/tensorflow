import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, mnist

import matplotlib.pyplot as plt

enc_input = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, activation='relu')(enc_input)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Flatten()(x)
enc_output = Dense(8, activation='linear')(x)

encoder = keras.Model(enc_input, enc_output, name="encoder")

dec_input = keras.Input(shape=(8,), name="encoded_img")
x = Dense(7 * 7 * 8, activation='relu')(dec_input)
x = keras.layers.Reshape((7, 7, 8))(x)
x = Conv2DTranspose(64, 5, strides=(2, 2), activation="relu", padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = Conv2DTranspose(32, 5, strides=(2, 2), activation="linear", padding='same')(x)
x = keras.layers.BatchNormalization()(x)
dec_output = Conv2DTranspose(1, 3, activation="sigmoid", padding='same')(x)

decoder = keras.Model(dec_input, dec_output, name="decoder")

autoencoder_input = Input(shape=(28, 28, 1), name="img")
x = encoder(autoencoder_input)
autoencoder_output = decoder(x)

autoencoder = keras.Model(autoencoder_input, autoencoder_output, name="autoencoder")
#autoencoder.summary()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, batch_size=32, epochs=1)

h = encoder.predict(tf.expand_dims(x_test[0], axis=0))
img = decoder.predict(h)

plt.subplot(121)
plt.imshow(x_test[0], cmap='gray')
plt.subplot(122)
plt.imshow(img.squeeze(), cmap='gray')
plt.show()
