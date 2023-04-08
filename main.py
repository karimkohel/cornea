import cv2
from corneaReader import CorneaReader
import numpy as np

cap = cv2.VideoCapture(0)
cornea = CorneaReader()

while True:

    ret, frame = cap.read()

    if ret:
        frame = cornea.readEyes(frame)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
np.save("data/test1", cornea.data)