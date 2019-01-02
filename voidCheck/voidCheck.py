# Imports
from __future__ import print_function
# import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read and show image
img = cv.imread('roentgen/test.gif')
# (h, w, d) = img.shape
# print("width={}, height={}, depth={}".format(w, h, d))


# Turn image into grayscale
def toGray(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img_gray


# Turn image into grayscale
def toBinary(img):
    img_binary = cv.threshold(img, 190, 255, cv.THRESH_BINARY)[1]
    return img_binary


# Turn image into grayscale
def toInvBinary(img):
    img_invbinary = cv.threshold(img, 190, 255, cv.THRESH_BINARY_INV)[1]
    return img_invbinary


# Search nutzens
def searchNutzens(img):
    # Find and draw contours around rectangles
    contours = cv.findContours(toInvBinary(toGray(img)), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1]

    area_list = []
    for i in contours:
        area = cv.contourArea(i)
        area = round(area)
        area_list.append(area)

    maxarea = max(area_list)
    good_list = []
    for index, area in enumerate(area_list):
        if area > maxarea * .8:
            good_list.append(index)

    cons = [i for n, i in enumerate(contours) if n in good_list]
    #cv.drawContours(img, cons, -1, (255, 0, 0), 1)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    cv.drawContours(mask, cons, -1, 0, -1)
    mask = toInvBinary(mask)
    masked_img = cv.bitwise_and(img, img, mask=mask)

    nutzens = cv.equalizeHist(toGray(masked_img))

    return nutzens


img2 = searchNutzens(img)
img2 = cv.threshold(img2, 122, 255, cv.THRESH_BINARY)[1]

contours2 = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1]
cv.drawContours(img2, contours2, 11, (255, 0, 0), 1)

# cv.imshow('Original image', img)
cv.imshow('Processed image', img)

cv.waitKey(0)
cv.destroyAllWindows()
