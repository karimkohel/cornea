import cv2
from classes.corneaReader import CorneaReader
import numpy as np

cap = cv2.VideoCapture(0)
cornea = CorneaReader()

while True:

    ret, frame = cap.read()

    if ret:
        _, frame = cornea.readEyes(frame, 'initTest')

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()

# TODO: Vstack metrics on frame array