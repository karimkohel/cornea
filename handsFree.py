import numpy as np
import tensorflow as tf
import pyautogui
import cv2
from classes.cornea import CorneaReader
import time

cap = cv2.VideoCapture(0)
cornea = CorneaReader()
model = tf.keras.models.load_model('models/convModelTest1.h5')


for i in range(50):
    _, frame = cap.read()


while True:


    inputFrames = []
    inputMetrics = []
    i=0
    while i<3:
        i = i + 1
        ret, frame = cap.read()
        (eyeMetrics, inputFrame), frame = cornea.readEyes(frame)
        inputFrames.append(cornea.preProcessOnTheFly(frame))
        inputMetrics.append(eyeMetrics)

    inputMetrics = np.array(inputMetrics)
    inputFrames = np.array(inputFrames)
    predictions = model.predict([inputFrames, inputMetrics])
    coordinates = np.average(predictions, axis=0)
    print(coordinates)
    pyautogui.moveTo(coordinates[0], coordinates[1])

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Frame", frame)
    

cap.release()
cv2.destroyAllWindows()
