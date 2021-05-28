import numpy as np
import cv2

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w//2, h//2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def resize(image, newWidth = None, newHeight = None, inter = cv2.INTER_AREA):
    #If conflicting width and height provided, the heigh will be used.
    height = image.shape[0]
    width = image.shape[1]

    if newWidth is None and newHeight is None:
        return image

    if newHeight is None:
        r = newWidth / width
        dim = (int(newWidth), int(int(height * newWidth) / int(width)))
    else:
        r = newHeight / height
        dim = (int(int(width * newHeight) / int(height)), int(newHeight))

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def flip(image, mode):
    return cv2.flip(image, mode)

def crop(image, startCoord, endCoord):  #Inputs are 2 value tuples
    return image[startCoord[1]:endCoord[1], startCoord[0]:endCoord[0]]

def changeBrightness(image, amount):
    M = np.ones(image.shape, dtype = "uint8") * abs(amount)
    if amount >= 0:
        return cv2.add(image, M)
    else:
        return cv2.subtract(image, M)
