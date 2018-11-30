# Imports
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

imgs = []
path = input("Give the path where the x-ray images are:")
extensions = [".jpg", ".png", ".tga"]

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(tuple(extensions)):
            imgs.append(os.path.join(root, file))

print("Images read: " + str(imgs.__len__()))
#print(*imgs, sep="\n")

img = cv.imread(imgs[0], 0)
equ = cv.equalizeHist(img)
res = np.hstack((img, equ)) #stacking images side-by-side
cv.imshow('res.png', res)
#cv.imwrite('res.png', res)
cv.waitKey(0)
cv.destroyAllWindows()
