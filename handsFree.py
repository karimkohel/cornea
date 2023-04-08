import numpy as np
import tensorflow as tf
import pyautogui
import time

data = np.load('data/test1.npy')

x = data[:, :33]
y = data[:, 33:]

model = tf.keras.models.load_model('trial1Model.h5')


predictions = model.predict(x)

for prediction in predictions:
    pyautogui.moveTo(prediction[0], prediction[1])
