import cv2
import mediapipe as mp
import numpy as np
from operator import rshift
mp_face_mesh = mp.solutions.face_mesh


LEFT_EYE=[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS=[474,475,476,477]
RIGHT_IRIS=[469,470,471,472]

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as faceMesh:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgHeigh, imgWidth = frame.shape[:2]
        results = faceMesh.process(frameRGB)

        if results.multi_face_landmarks:
            meshPoints = np.array([np.multiply([p.x, p.y], [imgWidth, imgHeigh]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            (leftCX, leftCY), leftRadius = cv2.minEnclosingCircle(meshPoints[LEFT_IRIS])
            (rightCX, rightCY), rightRadius = cv2.minEnclosingCircle(meshPoints[RIGHT_IRIS])
            leftCenter = np.array([leftCX, leftCY], dtype=np.int32)
            rightCenter = np.array([rightCX, rightCY], dtype=np.int32)

            cv2.circle(frame, leftCenter, int(leftRadius), (0,0,255), 1, cv2.LINE_AA)
            cv2.circle(frame, rightCenter, int(rightRadius), (0,0,255), 1, cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()