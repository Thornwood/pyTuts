# Imports
from __future__ import print_function
# import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read and show image
img = cv.imread('roentgen/test.jpg')
rows, cols, dims = img.shape
# (h, w, d) = img.shape
# print("width={}, height={}, depth={}".format(w, h, d))
# img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)

# Turn image into grayscale
def toGray(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img_gray

# Turn image into Blurred
def toBlur(img):
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    return img_blur

def toMBlur(img):
    median = cv.medianBlur(img, 5)
    return median

blur = cv.GaussianBlur(img, (3, 3), 0)
median = cv.medianBlur(img, 5)
# bilateral = cv.bilateralFilter(img, 9, 75, 75)
# imageIntegral = cv.integral(img)
hist = cv.calcHist([img], [0], None, [256], [0, 255])
plt.plot(hist)

min_tresh = 35
max_tresh = 150

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_binary = cv.threshold(img_gray, min_tresh, max_tresh, cv.THRESH_BINARY)[1]
img_invbinary = cv.threshold(img_gray, min_tresh, max_tresh, cv.THRESH_BINARY_INV)[1]
otsu = cv.threshold(img_gray, min_tresh, max_tresh, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
otsuinv = cv.threshold(img_gray, min_tresh, max_tresh, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]

_, contours, hierarchy = cv.findContours(img_invbinary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
