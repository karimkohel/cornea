import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


convInput = tf.keras.layers.Input(shape=(30,30, 1))
denseInput = tf.keras.layers.Input(shape=(33))

x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=convInput.shape)(convInput)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=x.shape)(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.concatenate([x, denseInput])
x = tf.keras.layers.Dense(160, activation='relu')(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='relu')(x)

model = tf.keras.Model(inputs=[convInput, denseInput], outputs=output, name='nesqafe2x1')
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.summary()