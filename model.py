import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


data = np.load('data/trial1.npy')

x = data[:, :33]
y = data[:, 33:]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, random_state=30)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(40, activation='relu', input_dim=33),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(2, activation='relu')
])

