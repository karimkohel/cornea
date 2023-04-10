import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


data = np.load('data/osamaTestfelBeit.npy')

x = data[50:, :33]
y = data[50:, 33:]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=30)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(40, activation='relu', input_dim=33),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu')
])

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history = model.fit(xTrain, yTrain, epochs=70, verbose=1, batch_size=24, validation_data=(xTest, yTest))
model.save("models/laptopModel2.h5")