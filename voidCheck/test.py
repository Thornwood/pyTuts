# Imports
from __future__ import print_function
# import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
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
'''

# If the image is color image
# if len(img.shape) == 3:
#   print("Color image: True")

# Read and show image
# img = cv.imread('roentgen/test.gif')
# (h, w, d) = img.shape
# print("width={}, height={}, depth={}".format(w, h, d))
# cv.imshow('img', img)


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

'''
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
'''


'''
# Set up the detector with default parameters.
detector = cv.SimpleBlobDetector()
# Create a detector with the parameters
ver = (cv.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv.SimpleBlobDetector()
else:
    detector = cv.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                      cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv.imshow("Keypoints", im_with_keypoints)
'''

'''

img = cv.imread('roentgen/test.jpg')
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
K2 = 4
ret2, label2, center2 = cv.kmeans(Z, K2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)

center2 = np.uint8(center2)
res3 = center2[label2.flatten()]
res4 = res3.reshape(img.shape)

substract = res4 - res2

cv.imshow('res2', res2)
cv.imshow('res4', res4)
cv.imshow('subs', substract)
'''


########################################################

'''
# Read and show image
img = cv.imread('roentgen/test.jpg')
rows, cols, dims = img.shape
# (h, w, d) = img.shape
# print("width={}, height={}, depth={}".format(w, h, d))
# img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)

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
'''

# Blob detection
'''
# Read image
im = cv.imread("roentgen/test.jpg", cv.IMREAD_GRAYSCALE)


# Create a detector with the parameters
detector = cv.SimpleBlobDetector_create()


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv.imshow("Keypoints", im_with_keypoints)
'''




cv.waitKey(0)
cv.destroyAllWindows()
