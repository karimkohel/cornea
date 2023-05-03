import cv2
from classes.cornea import CorneaReader
import numpy as np
import time 

cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cornea = CorneaReader(cap)

print("started")


ret, frameOrg = cap.read()


if ret:
    _, upFrame = cornea.readEyes(frameOrg, 'metricsTest')

    downFrame = cv2.pyrDown(frameOrg)
    _, downFrame = cornea.readEyes(downFrame, 'metricsTest')
    cv2.imshow("original frame ", frameOrg)

    cv2.imshow("Frame", upFrame)
    cv2.imshow("downFrame", downFrame)
    cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()

# import numpy as np
 
# # initializing points in
# # numpy arrays
# point1 = np.array((1, 2))
# point2 = np.array((1, 1))
 
# # calculating Euclidean distance
# # using linalg.norm()
# dist = np.linalg.norm(point1 - point2)
 
# # printing Euclidean distance
# print(dist)