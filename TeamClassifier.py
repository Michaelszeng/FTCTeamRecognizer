"""
This is the main CNN made using tensorflow to train a model to recognize photos
of the FTC team.

Windows 10 Laptop

CPU: 16 GB RAM, Intel(R) Core(TM) i7-7560U CPU @ 2.40 GHz
GPU: Intel(R) Iris(R) Plus Graphics 640
"""

# Python 3.8.6
# tensorflow 2.4.0
# matplotlib 3.3.3
# numpy 1.19.4
# opencv-python 4.4.0

import time
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets as datasets
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random
import pickle
import os
import imutils
from datetime import datetime
from collections import namedtuple
import pandas as pd
import pathlib
import IPython.display as display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_EPOCHS = 50
SHUFFLE = False

SAVE_MODEL = True
USE_SAVED_MODEL = True

BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

DATADIR = "C:\\Users\\Michael Zeng\\Documents\\Programming\\Recognizing FTC Team\\Training Images"
CATEGORIES = ["teamTrue", "teamFalse"]
training_data = []

def generatorXY(batch_size, data_set_list):
    #Each time the generator is called it furthers the loop through the dataset list and yields a batch; then when it is called again it yields the next batch.
    counter = 0
    while True:
        batchX, batchY = [], [] #Construct list for the newest batch
        for i in range(batch_size):
            if counter >= len(data_set_list[0]):
                counter = 0
            batchX.append(data_set_list[0][counter])    #Get the batch's data
            batchY.append(data_set_list[1][counter])
            counter += 1
        yield np.array(batchX), np.array(batchY)    #yield instead of return so the function remembers where it is in the loop next time it is called

def scheduler(epoch, lr):
    """
    Exponentially Decreasing Learning Rate Scheduler
    """
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

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
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
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
                if class_num == 0:
                    training_data.append([new_array, [1, 0]])
                else:
                    training_data.append([new_array, [0, 1]])
            except Exception as e:
                print(e)

fout = open('test.txt', 'w')
now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)
print("[TIMER] Process Time:", now, file = fout, flush = True)

try:
    loadedModel = models.load_model('models/model.h5')
    # print(loadedModel.summary())
    # print(loadedModel.get_weights())
    loadSuccessful = True
except:
    loadSuccessful = False
print("load successful: " + str(loadSuccessful))

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print('[INFO] GPU is detected.')
    print('[INFO] GPU is detected.', file = fout, flush = True)
else:
    print('[INFO] GPU not detected.')
    print('[INFO] GPU not detected.', file = fout, flush = True)
print('[INFO] Done importing packages.')
print('[INFO] Done importing packages.', file = fout, flush = True)

class Net():
    def __init__(self, input_shape):
        #Starting an empty model
        self.model = models.Sequential()

        #Input shape: 200x200

        # Conv2d() Parameters: Outgoing Layers, Frame Size
        # Conv2d() Keyword Parameters: strides (default is strides=1), padding (="valid" means 0, ="same" means whatever gives same output width/height as input).
        self.model.add(layers.Conv2D(6, 11, strides=3, input_shape = input_shape, activation = 'relu')) #200x200 --> 64x64
        self.model.add(layers.MaxPooling2D(pool_size = 2))  #64x64 --> 32x32
        self.model.add(layers.Dropout(0.05))
        self.model.add(layers.Conv2D(12, 3, strides=1, input_shape = input_shape, activation = 'relu')) #32x32 --> 30x30
        self.model.add(layers.MaxPooling2D(pool_size = 3))  #30x30 --> 10x10
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(24, 3, strides=1, input_shape = input_shape, activation = 'relu')) #10x10 --> 8x8
        self.model.add(layers.MaxPooling2D(pool_size = 2))  #8x8 --> 4x4
        self.model.add(layers.Dropout(0.15))
        # 384 output variables

        self.model.add(layers.Flatten())

        # Parameters: Outgoing Layers, Activation Func.
        self.model.add(layers.Dense(96, activation = 'relu'))
        self.model.add(layers.Dense(24, activation = 'relu'))
        self.model.add(layers.Dense(2))

        #Keyword Parameters: lr (learning rate), momentum
        # self.optimizer = optimizers.SGD(lr=0.001, momentum=0.9)
        self.optimizer = optimizers.Adam(lr=0.001)
        # self.loss = losses.MeanSquaredError()
        self.loss = losses.BinaryCrossentropy()
        #Keyword Parameters: loss (the loss function to use), optimizer (the optimizer to use), metrics (what metrics it will measure and output)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)
        print(summaryStr, file=fout)

print("[INFO] Loading Training and Test Datasets.")
print("[INFO] Loading Training and Test Datasets.", file=fout)

#importing the dataset:
create_training_data()
random.shuffle(training_data)

X = []  #this array contains all the images
Y = []  #this array contains all the respective labels

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, 200, 200, 3)

#Saving the dataset
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("Y.pickle", "rb")
Y = pickle.load(pickle_in)

#Testing labeling
# cv2.imshow("img", X[0])
# print(Y[0])
# cv2.waitKey(0)
# cv2.imshow("img", X[1])
# print(Y[1])
# cv2.waitKey(0)
# cv2.imshow("img", X[2])
# print(Y[2])
# cv2.waitKey(0)
# cv2.imshow("img", X[3])
# print(Y[3])
# cv2.waitKey(0)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.25, shuffle = True)
print("[INFO] Done creating training and testing datasets.\n")

# Convert labels from integers to vectors.
lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print(testX)
print(testY)

classes = ["teamTrue", "teamFalse"]

net = Net((200, 200, 3))
# Notice that this will print both to console and to file.
print(net)

if loadSuccessful == False or USE_SAVED_MODEL == False: #if there is no saved model or if I want to re-train the model
    print("Training New Model")
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    results = net.model.fit(x=generatorXY(BATCH_SIZE_TRAIN, [trainX, trainY]), validation_data=generatorXY(BATCH_SIZE_TEST, [testX, testY]), shuffle = SHUFFLE, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, callbacks = [callback], verbose = 1, steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN, validation_steps=len(testX)/BATCH_SIZE_TEST)
    # results = net.model.fit(trainX, trainY, validation_data=(testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1)
    plt.figure()
    plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'])
    plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'])
    plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['accuracy'])
    plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_accuracy'])
    plt.show()
else:   #if I want to use the existing model just to make predictions
    print("Using Old Model")
    results = loadedModel.predict(testX, batch_size=None, verbose=0,  steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    correct = 0
    for i in range(len(results)):
        cv2.imshow("X", testX[i])
        print("Y: " + str(np.argmax(testY[i])))
        # print(results[i])
        indices = np.argpartition(results[i], -2)[-1:]   #Getting unsorted list of indices of 1 largest value in result[i]
        # indices = indices[np.argsort(results[i][indices])]  #sorting the indices list by their respective values in results[i]
        # print(indices)
        if np.argmax(results[i]) == np.argmax(testY[i]):
            # print("Correct on " + str(i))
            correct += 1

    print(str(100*correct/len(results)) + "% Accuracy; " + str(correct) + " correct out of " + str(len(results)))
    # results = loadedModel.fit(trainX, trainY, validation_data=(testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1)

# File location to save to or load from
if os.path.isfile('models/model.h5') is False and SAVE_MODEL == True:  #only save the model if it's not already saved
    net.model.save('models/model.h5')   #use h5 extension
