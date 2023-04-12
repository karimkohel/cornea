import cv2
import mediapipe as mp
import numpy as np
import os
import pyautogui


class CorneaReader():
    """Class for reading the cornea location and deriving whatever values are needed from it
    """

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [133, 33, 7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_CENTER = 473
    RIGHT_IRIS_CENTER = 468

    EYESTRIP = [27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258]
    #     'leftLid': (257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258),
    #     'rightLid': (27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29),
    #     'eyeLids': (27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258)
    # }

    def __init__(self) -> None:
        """Start the facemesh solution and be ready to read eye values

        """
        mp_face_mesh = mp.solutions.face_mesh
        self.faceMesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.data = None

    def readEyes(self, frame: np.ndarray, saveDir: str = None) -> np.ndarray:
        """Method to derive all eye points needed from a frame

        the method saves the eye distances as the first 33 elements of the data array, the xy labels for mouse as the 34th and 35th elements, while the rest is the image data

        Input:
        -------
        frame: required, numpy array representing the camera frame

        Returns:
        ---------
        frame: the same input frame after processing and applying shapes to visualize points
        leftIrisDistances: a numpy array containing the Euclidean distances from the left eye iris to all left eye points
        rightIrisDistances: a numpy array containing the Euclidean distances from the right eye iris to all right eye points
        middleEyeDistance: the distance between the closest points of each eye

        Output Format:
        --------------
        frame, (left, right, middle)
        """
        mousePos = pyautogui.position()
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgHeigh, imgWidth = frame.shape[:2]
        results = self.faceMesh.process(frameRGB)

        if results.multi_face_landmarks:
            meshPoints = np.array([np.multiply([p.x, p.y], [imgWidth, imgHeigh]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            frame = self.__visualize(frame, meshPoints, meshPoints[self.LEFT_IRIS_CENTER], meshPoints[self.RIGHT_IRIS_CENTER])
            croppedFrame = self.__cropEye(frame, meshPoints)

            leftIrisDistances = np.linalg.norm(meshPoints[self.LEFT_EYE] - meshPoints[self.LEFT_IRIS_CENTER], axis=1)
            rightIrisDistances = np.linalg.norm(meshPoints[self.RIGHT_EYE] - meshPoints[self.RIGHT_IRIS_CENTER], axis=1)
            middleEyeDistance = np.linalg.norm(meshPoints[self.RIGHT_EYE[0]] - meshPoints[self.LEFT_EYE[0]])
            allMetrics = np.concatenate((leftIrisDistances, rightIrisDistances, [middleEyeDistance, mousePos[0], mousePos[1]]))
            allData = np.concatenate((allMetrics, croppedFrame))

            if saveDir:
                self.__saveDataArray(allData, saveDir)

            return allData, frame

        return None, frame

    def __visualize(self, frame: np.ndarray, meshPoints: np.ndarray, leftCenter: np.ndarray, rightCenter: np.ndarray) -> np.ndarray:
        """private method to visualize gathered eye data on the current frame"""

        cv2.circle(frame, leftCenter, 1, (0,255,0), 3)
        cv2.circle(frame, rightCenter, 1, (0,255,0), 3)

        return frame

    def __cropEye(self, frame: np.ndarray, meshPoints: np.ndarray) -> np.ndarray:
        eyeStripCoordinates = meshPoints[self.EYESTRIP]
        maxX, maxY = np.amax(eyeStripCoordinates, axis=0)
        minX, minY = np.amin(eyeStripCoordinates, axis=0)
        frame = frame[minY:maxY, minX:maxX]
        return frame


    def __saveDataArray(self, dataArray: np.ndarray, saveDir: str) -> None:
        i = 0
        try:
            prevDataFiles = os.listdir(f"data/{saveDir}")
            i = len(prevDataFiles)
        except FileNotFoundError:
            os.mkdir(f"data/{saveDir}")
        np.save(f"data/{saveDir}/{i}", dataArray)


    def __del__(self) -> None:
        self.faceMesh.close()