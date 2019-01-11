# Imports
from __future__ import print_function
# import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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
def to_blur(image):
    image = cv.GaussianBlur(image, (3, 3), 0)
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


# Read and show image
img = cv.imread('roentgen/test.jpg')
rows, cols, dims = img.shape
# (h, w, d) = img.shape
# print("width={}, height={}, depth={}".format(w, h, d))
# img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)

blur = 5
min_tresh = 35
max_tresh = 150

plt.plot(to_gray(calchist(img)))

_, contours, hierarchy = cv.findContours(to_invbinary(img, min_tresh, max_tresh), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

cv.drawContours(img, contours, 0, (255, 0, 0), 1)

# For each contour, find the bounding rectangle and draw it
for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]
    x, y, w, h = cv.boundingRect(currentContour)
    if currentHierarchy[3] < 0:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)  # these are the innermost child components
#   elif currentHierarchy[2] < 0:
#       cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)  # these are the outermost parent components

cv.imshow('Original image', img)
# cv.imshow('Gray', img_gray)
# cv.imshow('Binary', img_binary)
# cv.imshow('Invert Binary', img_invbinary)
# cv.imshow('Otsu', otsu)
# cv.imshow('Invert Otsu', otsuinv)
# cv.imshow('Gaussian Blurred', blur)
# cv.imshow('Median Blurred', median)
# cv.imshow('Bilateral Blurred', bilateral)
# plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
