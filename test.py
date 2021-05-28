"""
This program is a test for creating the training and testing data sets. It uses the
cropping algorithm in imageCropTest.py and takes photos from the Training Images folder
"""

import numpy as np
import argparse
import cv2
import os
from pathlib import Path
import imutils
import tensorflow.keras.datasets as datasets
from datetime import datetime
from collections import namedtuple
import pandas as pd
import pathlib
import IPython.display as display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import pickle
#
# data_directory = pathlib.WindowsPath("c:/Documents/Programming/Recognizing FTC Team/Training Images/")
# CLASSES = np.array([item.name for item in data_directory.glob('*') if item.name != "LICENSE.txt"])
#
# image_generator = ImageDataGenerator(rescale=1./255)
#
# dataset = image_generator.flow_from_directory(directory=str(data_directory), batch_size=32, shuffle=True, target_size=(300, 500), classes = list(CLASSES))



DATADIR = "C:\\Users\\Michael Zeng\\Documents\\Programming\\Recognizing FTC Team\\Training Images"
CATEGORIES = ["teamTrue", "teamFalse"]
training_data = []

def imageManipulation(image):
    image = imutils.resize(image, newHeight = 500)

    #cropping the image so it is always the same aspect ratio
    height, width, channels = image.shape
    normalAspectRatio = 1
    if height > width:
        croppedImageHeight = int(width / normalAspectRatio)
        heightCropStart = int((height - croppedImageHeight)/2)
        heightCropEnd = int(((height - croppedImageHeight)/2) + croppedImageHeight)
        image = imutils.crop(image, (0, heightCropStart), (width, heightCropEnd))
    else:
        croppedImageWidth = int(height / normalAspectRatio)
        widthCropStart = int((width - croppedImageWidth)/2)
        widthCropEnd = int(((width - croppedImageWidth)/2) + croppedImageWidth)
        image = imutils.crop(image, (widthCropStart, 0), (widthCropEnd, height))

    #resizing to reduce resolution to decrease run time
    image = imutils.resize(image, newHeight = 200)
    return image

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                #do whatever image manipulation stuff is necessary
                new_array = imageManipulation(img_array)
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)

create_training_data()
random.shuffle(training_data)

X = []  #this array contains all the images
Y = []  #this array contains all the respective labels

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, 200, 200, 3)

print(X)
print(Y)

#Saving the dataset
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

#Procedure for importing the saved dataset:
# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)
# pickle_in = open("Y.pickle", "rb")
# Y = pickle.load(pickle_in)
