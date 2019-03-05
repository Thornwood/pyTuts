# Imports
from __future__ import print_function
# import os
import cv2 as cv

# Values
blur = 15
min_tresh = 116
max_tresh = 255


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
def to_mblur(image, _blur):
    image = cv.medianBlur(image, _blur)
    return image


# Calculate Image Histogram
def calchist(image):
    image = cv.calcHist([image], [0], None, [256], [0, 255])
    return image
# endregion


# Read image
img = cv.imread("roentgen/images/20190228132404_001_1_IO.jpg")
img = to_gray(img)

img2 = to_mblur(img, blur)
img3 = to_binary(img2, min_tresh, max_tresh)
img4 = to_otsu(img3, min_tresh, max_tresh)

cv.imshow('Image', img2)
cv.imshow('Image4', img4)

cv.waitKey(0)
cv.destroyAllWindows()
