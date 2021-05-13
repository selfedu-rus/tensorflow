import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super().__init__()
        self.units = units
        self.rate = 0.01

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        regular = 100.0 * tf.reduce_mean(tf.square(self.w))
        self.add_loss(regular)
        self.add_metric(regular, name="mean square weights")

        return tf.matmul(inputs, self.w) + self.b


class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseLayer(128)
        self.layer_2 = DenseLayer(10)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.softmax(x)
        return x


model = NeuralNetwork()

# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
#              loss=tf.losses.categorical_crossentropy,
#              metrics=['accuracy'])

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model.fit(x_train, y_train, batch_size=32, epochs=5)

print( model.evaluate(x_test, y_test_cat) )
