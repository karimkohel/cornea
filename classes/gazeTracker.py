from .cornea import CorneaReader
import cv2
from tensorflow.keras.models import load_model as tf_load_model
import numpy as np
from typing import Tuple


class GazeTracker():
    def __init__(self,  modelDir: str) -> None:
        """Constructor method for gaze tracker to find and load model and start cornea reader
        
        Input:
        --------
        modelDir: the directory and name of the model to be used in predictions
        """
        self.corneaReader = CorneaReader()
        self.model = tf_load_model(modelDir)


    def track_gaze(self, frames: list[np.ndarray]) -> np.ndarray:
        """Main method for taking in a list of frames and then giving the output coordinates of user gaze on the screen.
            the method will loop over the given frames and predict gaze location for each one, then average out the predictions and return the result

        Input:
        ---------
        frames: List of frames containing at least one opencv cap frame
        """

        inputFrames = []
        inputMetrics = []

        for frame in frames:
            (eyeMetrics, inputFrame), frame = self.corneaReader.readEyes(frame)
            inputFrames.append(self.corneaReader.preProcessOnTheFly(frame))
            inputMetrics.append(eyeMetrics)

        inputMetrics = np.array(inputMetrics)
        inputFrames = np.array(inputFrames)
        predictions = self.model.predict([inputFrames, inputMetrics])
        coordinates = np.average(predictions, axis=0)

        return coordinates