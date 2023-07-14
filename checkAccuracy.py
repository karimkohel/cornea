import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
from classes.gazeTracker import GazeTracker
import statistics

cap = cv2.VideoCapture(0)
gazeTracker = GazeTracker('models/convModelTest8.h5')

for i in range(30):
    _, frame = cap.read()

distances = []


for i in range(1500):

    frames = []
    for _ in range(1):
        ret, frame = cap.read()
        frames.append(frame)

    predictedCoordinates = gazeTracker.track_gaze(frames)
    realCoordinates = np.array(pyautogui.position())

    dist = np.linalg.norm(predictedCoordinates - realCoordinates, axis=0)
    distances.append(int(dist))


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Frame", frame)
    


cap.release()
cv2.destroyAllWindows()

plt.hist(distances, bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.5)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

distancesNpArray = np.array(distances)
stdDeviation = np.std(distancesNpArray, ddof=1)

print("Mean: ", statistics.mean(distances))
print("Standard Deviation: ", stdDeviation)