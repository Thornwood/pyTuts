# Imports
from __future__ import print_function
# import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
imgs = []
path = input("Give the path where the x-ray images are:")
extensions = [".jpg", ".png", ".tga"]

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(tuple(extensions)):
            imgs.append(os.path.join(root, file))

print("Images read: " + str(imgs.__len__()))
# print(*imgs, sep="\n")

img = cv.imread(imgs[0], 0)
equ = cv.equalizeHist(img)
res = np.hstack((img, equ)) # stacking images side-by-side
cv.imshow('res.png', res)
# cv.imwrite('res.png', res)
"""


# Read and show image
# img = cv.imread('roentgen/test.gif')
# (h, w, d) = img.shape
# print("width={}, height={}, depth={}".format(w, h, d))
# cv.imshow('img', img)

# Turn image to grayscale
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Must needed for turn to binary
# img_gray = np.float32(img_gray)
# cv.imshow('img_gray', img_gray)

# Image turn to Binary
# img_binary = cv.threshold(img_gray, 190, 255, cv.THRESH_BINARY)[1]
# cv.imshow('img_binary', img_binary)

# Image turn to Binary invert
# img_binary_inv = cv.threshold(img_gray, 190, 255, cv.THRESH_BINARY_INV)[1]
# cv.imshow('img_binary_inv', img_binary_inv)

''' # This below is working
# global thresholding
ret1,th1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img_gray,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
'''
'''
img_masked = cv.bitwise_and(img, img, mask=img_binary_inv)
cv.imshow('masked', img_masked)

equ = cv.equalizeHist(img_gray)
# cv.imshow('Equalized image', equ)

blur = cv.GaussianBlur(equ, (5, 5), 0)
# cv.imshow('blur', blur)

ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('tresh', th3)

hist = cv.calcHist([img_masked], [0], None, [256], [0, 256])
plt.hist(img_masked.ravel(), 256, [0, 256])
plt.title('Histogram for gray scale picture')
plt.show()
'''



cv.waitKey(0)
cv.destroyAllWindows()


