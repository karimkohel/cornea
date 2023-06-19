import cv2
import pyautogui
from classes.gazeTracker import GazeTracker

cap = cv2.VideoCapture(0)
gazeTracker = GazeTracker('models/convModelTest3.h5')

for i in range(30):
    _, frame = cap.read()


while True:

    frames = []
    for _ in range(3):
        ret, frame = cap.read()
        frames.append(frame)

    coordinates = gazeTracker.track_gaze(frames)

    pyautogui.moveTo(coordinates[0], coordinates[1])

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Frame", frame)
    

cap.release()
cv2.destroyAllWindows()
