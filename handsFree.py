import numpy as np
import tensorflow as tf
import pyautogui
import time

data = np.load('data/test2.npy')

x = data[60:, :33]
y = data[60:, 33:]

model = tf.keras.models.load_model('trial1Model.h5')


# for i in range(start=0, stop=len(x), step=10):
#     try:




predictions = model.predict(x)

for i in range(0, len(predictions), 10):
    try:
        values = predictions[i:i+10]
        coordinates = np.average(values, axis=0)
        pyautogui.moveTo(coordinates[0], coordinates[1])
    except Exception:
        break
