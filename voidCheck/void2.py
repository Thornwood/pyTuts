# Imports
from __future__ import print_function
# import os
import cv2 as cv
import numpy as np

# Values
blur = 15
min_tresh = 118
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


def w_bilateral(image):
    image = cv.bilateralFilter(image, 50, 10, 10, None, 1)
    return image
# endregion


# Read image
img = cv.imread("roentgen/images/20190228132404_001_1_IO.jpg")
gray = to_gray(img)
gray = cv.bilateralFilter(gray, 100, 10, 150)

ret, thresh = cv.threshold(gray, 118, 255, 0)

_, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

'''
for cnt in contours:
    approx = cv.approxPolyDP(cnt, 0.001 * cv.arcLength(cnt, True), True)
    cv.drawContours(img, [approx], 0, (0, 255, 255), 1)
'''

cv.drawContours(img, contours, 0, (255, 0, 0), 1)

# For each contour, find the bounding rectangle and draw it
for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]
    x, y, w, h = cv.boundingRect(currentContour)
    if currentHierarchy[3] < 0:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)  # these are the innermost child components
    elif currentHierarchy[2] < 0:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)  # these are the outermost parent components

cv.imshow('Image', img)

cv.waitKey(0)
cv.destroyAllWindows()
