import cv2
import mediapipe as mp
import numpy as np
from operator import rshift
mp_face_mesh = mp.solutions.face_mesh

class CorneaReader():
    """Class for reading the cornea location and deriving whatever values are needed from it
    """
    def __init__(self) -> None:
        pass
