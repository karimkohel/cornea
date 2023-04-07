import cv2
from corneaReader import CorneaReader

LEFT_EYE=[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS=[474,475,476,477]
RIGHT_IRIS=[469,470,471,472]

cap = cv2.VideoCapture(0)
cornea = CorneaReader()

while True:

    ret, frame = cap.read()

    if ret:
        frame = cornea.readEyes(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()