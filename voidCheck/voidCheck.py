# Imports
from __future__ import print_function
# import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Values
blur = 5
min_tresh = 35
max_tresh = 150


# region Image convert functions
# Convert image to Gray
def to_gray(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


# Convert image to Binary
def to_binary(image, min_tresh, max_tresh):
    image = cv.threshold(image, min_tresh, max_tresh, cv.THRESH_BINARY)[1]
    return image


# Convert image to Invert Binary
def to_invbinary(image, min_tresh, max_tresh):
    image = cv.threshold(image, min_tresh, max_tresh, cv.THRESH_BINARY_INV)[1]
    return image


# Convert image to Otsu
def to_otsu(image, min_tresh, max_tresh):
    image = cv.threshold(image, min_tresh, max_tresh, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    return image


# Convert image to Invert Otsu
def to_invotsu(image, min_tresh, max_tresh):
    image = cv.threshold(image, min_tresh, max_tresh, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]
    return image


# Convert image to Blurred
def to_blur(image, _blur):
    image = cv.GaussianBlur(image, (_blur, _blur), 0)
    return image


# Convert image to Median Blurred
def to_mblur(image):
    image = cv.medianBlur(image, 5)
    return image


# Calculate Image Histogram
def calchist(image):
    image = cv.calcHist([image], [0], None, [256], [0, 255])
    return image
# endregion


# Read image
img = cv.imread("roentgen/images/20190228132404_001_1_IO.jpg")
img = to_gray(img)

hist = cv.calcHist([img], [0], None, [256], [0, 256])
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histogram for gray scale picture')
plt.show()

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
