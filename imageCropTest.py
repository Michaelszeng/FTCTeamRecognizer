"""
This program is a test for the cropping algorithm to create consistent image sizes
for training.
"""

import numpy as np
import argparse
import cv2
import os
from pathlib import Path
import imutils
from datetime import datetime
from collections import namedtuple
import pandas as pd

# image = cv2.imread("testImageHorizontal.jpg")
image = cv2.imread("testImageVertical.jpg")
image = imutils.resize(image, newHeight = 500)
cv2.imshow("image", image)

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
height, width, channels = image.shape
print("width: " + str(width) + "; height: " + str(height))

cv2.imshow("imagecropped", image)
cv2.waitKey(0)
