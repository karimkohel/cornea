import numpy as np
import tensorflow as tf
import pyautogui
import cv2
from corneaReader import CorneaReader

cap = cv2.VideoCapture(0)
cornea = CorneaReader()
model = tf.keras.models.load_model('models/laptopModel.h5')

while True:


    x = None
    i=0
    while i<10:
        ret, frame = cap.read()
        sample, frame = cornea.readEyes(frame)
        if type(sample) == np.ndarray:
            i = i + 1
            if type(x) != np.ndarray:
                x = sample[:33]
            else:
                x = np.vstack([x, sample[:33]])

    predictions = model.predict(x)
    coordinates = np.average(predictions, axis=0)
    print(coordinates)
    pyautogui.moveTo(coordinates[0], coordinates[1])

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Frame", frame)
    

cap.release()
cv2.destroyAllWindows()
