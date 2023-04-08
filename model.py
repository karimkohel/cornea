import tensorflow as tf
import pandas as pd
import numpy as np


data = np.load('data/trial1.npy')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(0,33)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10)
])