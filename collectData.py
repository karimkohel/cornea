import cv2
from classes.cornea import CorneaReader
import numpy as np

cap = cv2.VideoCapture(0)
cornea = CorneaReader()


print("started")

while True:

    ret, frame = cap.read()

    if ret:

        key = cv2.waitKey(1)
        if key == ord('s'):
            break
        elif key == ord('q'):
            exit()
        cv2.imshow("Frame", frame)

while True:

    ret, frame = cap.read()

    if ret:
        _, croppedFrame = cornea.readEyes(frame, 'continuedDateTest')

        key = cv2.waitKey(3)
        if key == ord('q'):
            break
        cv2.imshow("cropped frame", croppedFrame)
        cv2.imshow("Frame", frame)


cap.release()
cv2.destroyAllWindows()
