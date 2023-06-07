import cv2
import mediapipe as mp
import numpy as np
import os, time
import pyautogui
from numpy import savez_compressed


class CorneaReader():
    """Class for reading the cornea location and deriving whatever values are needed from it
    """

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [133, 33, 7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS_CENTER = 473
    RIGHT_IRIS_CENTER = 468
    FACE_CENTER = 6
    
    FACE_METRICS_LEN = 37

    EYESTRIP = [27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258]
    TARGET_IMG_SIZE = [50, 120]

    def __init__(self) -> None:
        """Start the facemesh solution and be ready to read eye values & fetch eye images
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
        self.saveDir = saveDir
        mousePos = pyautogui.position()
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgHeigh, imgWidth = frame.shape[:2]
        results = self.faceMesh.process(frameRGB)
        self.CAMWIDTH, self.CAMHIGHT = frame.shape[0:2]

        if results.multi_face_landmarks:
            meshPoints = np.array([np.multiply([p.x, p.y], [imgWidth, imgHeigh]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # frame = self.__visualize(frame, meshPoints, meshPoints[self.LEFT_IRIS_CENTER], meshPoints[self.RIGHT_IRIS_CENTER]) # NOT FOR PRODUCTION
            ret, frame = self.__cropEye(frame, meshPoints)
            if not ret:
                print("unaccepted frame")
                return None, frame
            eyesMetrics = self.__calcEyeMetrics(meshPoints, frame)


            if saveDir:
                self.__saveDataArray(eyesMetrics, frame, mousePos, saveDir)

            return (eyesMetrics, frame), frame
        
        return None, frame

    def __visualize(self, frame: np.ndarray, meshPoints: np.ndarray, leftCenter: np.ndarray, rightCenter: np.ndarray) -> np.ndarray:
        """private method to visualize gathered eye data on the current frame"""

        cv2.circle(frame, leftCenter, 1, (0,255,0), 3)
        cv2.circle(frame, rightCenter, 1, (0,255,0), 3)

        return frame

    def __calcEyeMetrics(self, meshPoints: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Method to take in the meshpoints and calculate all the features we need from eye metrics as normalized eye distances from cornea"""


        leftIrisDistances = np.linalg.norm(meshPoints[self.LEFT_EYE] - meshPoints[self.LEFT_IRIS_CENTER], axis=1)
        rightIrisDistances = np.linalg.norm(meshPoints[self.RIGHT_EYE] - meshPoints[self.RIGHT_IRIS_CENTER], axis=1)
        middleEyeDistance = np.linalg.norm(meshPoints[self.RIGHT_EYE[0]] - meshPoints[self.LEFT_EYE[0]])
        centerToEdgesDistances = np.linalg.norm(meshPoints[self.FACE_CENTER] - np.array([(0, 0), (0, frame.shape[1]), (0, frame.shape[0]), (frame.shape[1], frame.shape[0])]), axis=1)

        eyesMetrics = np.concatenate((leftIrisDistances, rightIrisDistances, centerToEdgesDistances, [middleEyeDistance]))


        # for standard scaling later
        # fullCamScale = np.linalg.norm(np.array((0,0)) - np.array((self.CAMWIDTH, self.CAMHIGHT)))
        # ratios = leftIrisDistances/fullCamScale

        # print(leftIrisDistances[3])

        # print(fullCamScale)
        # print(ratios[3])

        return eyesMetrics

    def __cropEye(self, frame: np.ndarray, meshPoints: np.ndarray) -> np.ndarray:
        """private method to take in the entire frame and crop the eyestrip with max enclosure"""
        eyeStripCoordinates = meshPoints[self.EYESTRIP]
        maxX, maxY = np.amax(eyeStripCoordinates, axis=0)
        minX, minY = np.amin(eyeStripCoordinates, axis=0)
        frame = frame[minY:maxY, minX:maxX]
        if (frame.shape[:2][0] > self.TARGET_IMG_SIZE[0]) or (frame.shape[:2][1] > self.TARGET_IMG_SIZE[1]) or (frame.shape[:2][0] > frame.shape[:2][1]):
            return False, frame
        # print(frame.shape[:2])
        return True, frame


    def __saveDataArray(self, eyesMetrics: np.ndarray, croppedFrame: np.ndarray, mousePos: list[int], saveDir: str) -> None:
        """private method that would save data arrays to memory based on input to read eyes method and current data index"""
        try:
            prevDataFiles = os.listdir(f"data/{saveDir}")
            i = len(prevDataFiles)
        except FileNotFoundError:
            i = 0
            os.mkdir(f"data/{saveDir}")

        self.savedSampleCount = i
        np.savez(f"data/{saveDir}/{i}", eyesMetrics=eyesMetrics, croppedFrame=croppedFrame, mousePos=mousePos)

    def preProcess(self, dataDir: str = None):
        """static method to load the image and metrics data from a given directory"""
        if dataDir:
            filesPath = f"data/{dataDir}/"
            samplesFilesNames = os.listdir(filesPath)

            for i, file in enumerate(samplesFilesNames):
                file = np.load(filesPath+file)
                resizedFrame = self.__paddingRestOfImage(file['croppedFrame'])
                np.savez(f"data/{dataDir}/{i}", eyesMetrics=file['eyesMetrics'], croppedFrame=resizedFrame, mousePos=file['mousePos'])

        
    def loadData(self, dataDir: str = None, frame: np.ndarray = None) -> np.ndarray:
        if dataDir:
            filesPath = f"data/{dataDir}/"
            samplesFilesNames = os.listdir(filesPath)
            eyesMetrics = np.empty((len(samplesFilesNames), self.FACE_METRICS_LEN))
            frames = []
            mousePos = np.empty((len(samplesFilesNames), 2))

            for i, file in enumerate(samplesFilesNames):
                file = np.load(filesPath+file)
                eyesMetrics[i] = file['eyesMetrics']
                frames.append(file['croppedFrame'])                
                mousePos[i] = file['mousePos']
            frames = np.array(frames)
        
            return (eyesMetrics, frames, mousePos)
        else:
            return self.__paddingRestOfImage(frame)

    def __del__(self) -> None:
        """dunder delete method to clean up class after finishing"""
        self.faceMesh.close()


    def __resizeAspectRatio(self, image: np.ndarray) -> np.ndarray:
        """method to take input as a cropped frame with any dimension and then resize and zero fill it to the correct ration"""
        widthPerc = 0
        heightPerc = 0
        

        heightPerc = self.TARGET_IMG_SIZE[0] - image.shape[0]
        widthPerc = self.TARGET_IMG_SIZE[1] - image.shape[1]

        heightPerc = heightPerc * 100 / self.TARGET_IMG_SIZE[0]
        widthPerc = widthPerc * 100 / self.TARGET_IMG_SIZE[1]

        if heightPerc < widthPerc:
            scale_percent = heightPerc

        else:
            scale_percent = widthPerc        

        width = image.shape[1] + int(self.TARGET_IMG_SIZE[1] * scale_percent / 100)
        height = image.shape[0] + int(self.TARGET_IMG_SIZE[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        image = self.__paddingRestOfImage(image)   
        
        return image
    
    def __paddingRestOfImage(self, image: np.ndarray) -> np.ndarray:
        # Get the current size of the image
        current_size = image.shape[:2]

        # Compute the amount of padding needed
        padding_width = self.TARGET_IMG_SIZE[1] - current_size[1]
        padding_height = self.TARGET_IMG_SIZE[0] - current_size[0]
        
        image = cv2.copyMakeBorder(image,
                                top=0,
                                bottom=padding_height,
                                left=0,
                                right=padding_width,
                                borderType=cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
        return image

    def __showFrameThenExit(self, frame: np.ndarray, sec: int) -> None:
        """A debugging method used to show a frame for a number of seconds and exit do not use unless debugging only"""
        cv2.imshow("frame", frame)
        cv2.waitKey(sec*1000)
        cv2.destroyAllWindows()
        exit()


    def saveMetaData(self):
        with open(f"data/{self.saveDir}/sampleCount.txt", "w") as f:
            f.write(str(self.savedSampleCount))
