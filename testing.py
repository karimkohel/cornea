import cv2
from classes.cornea import CorneaReader
import numpy as np
import time 

cap = cv2.VideoCapture(0)
cornea = CorneaReader(cap)

print("started")


while True:
    ret, frameOrg = cap.read()
    if ret:
        _, upFrame = cornea.readEyes(frameOrg, 'metricsTest')

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        cv2.imshow("original frame ", upFrame)

cap.release()
cv2.destroyAllWindows()
