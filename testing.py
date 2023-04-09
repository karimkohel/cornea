import numpy as np
import tensorflow as tf
import pyautogui

data = np.load('data/cornersTest.npy')

x = data[60:, :33]
y = data[60:, 33:]

model = tf.keras.models.load_model('models/bigModel.h5')


predictions = model.predict(x)


for prediction in predictions:
    try:
        pyautogui.moveTo(prediction[0], prediction[1])
    except Exception as e:
        print(e)
        break


# for i in range(0, len(predictions), 10):
#     try:
#         values = predictions[i:i+10]
#         coordinates = np.average(values, axis=0)
#         pyautogui.moveTo(coordinates[0], coordinates[1])
#     except Exception:
#         break