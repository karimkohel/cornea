#this is pseudo code just thinking kda to scape form my inside 
#having file contain [(frame,array_of_points,mospos),(frame,array_of_points,mospos)]
#we need to create two approaches 1- make generators 2- use tf pipelining on each batch 
import cornea 
import tensorflow as tf 
import numpy as np
import cv2
import os



class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, dataDir: str, batch_size: int = 10000):

        self.dataDir = dataDir
        self.batch_size = batch_size

        with open(f"data/{self.dataDir}/sampleCount.txt", "r") as f:
            self.n = int(f.read())

    
    def __get_input(self, sample: int):

        file = np.load(f"data/{self.dataDir}/{sample}.npz")
        eyesMetrics = file['eyesMetrics']
        resizedFrame = self.__paddingRestOfImage(file['croppedFrame'])
        
        return (eyesMetrics, resizedFrame, mousePos)
    
    def __get_output(self, sample: int):
        file = np.load(f"data/{self.dataDir}/{sample}.npz")
        return file['mousePos']
    
    def __get_data(self, batches):

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, tuple([y0_batch, y1_batch])
    
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
    
    def __getitem__(self, index):
        print(index)
        exit(0)
        
        batches = self.df[index: next self.batchsize]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
