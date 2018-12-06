# OpenCV program to perform Edge detection in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2

# np is an alias pointing to numpy library
import numpy as np

# cv2.namedWindow("lll")

# capture framse from a camera
cap = cv2.VideoCapture(0)

while (cap.isOpened()):

    # reads frames from a camera
    ret, frame = cap.read()

    # convert colors from BGR to RGBA
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Display an original image
    # cv2.imshow("CAM Original", color)

    # Display grayscale image
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display in grayscale image
    cv2.imshow('Grayscale filter', grayscale)

    # finds edges in the input image image and
    # marks them in the output map edges
    cannyEdges = cv2.Canny(frame, 100, 200)
    cannyGray = cv2.Canny(grayscale, 100, 200)

    # Display edges in a frame
    # cv2.imshow('Canny from Original', cannyEdges)
    cv2.imshow('Canny from Grayscale', cannyGray)

    # Display color edges in a frame
    dst = cv2.bitwise_and(frame, frame, mask=cannyEdges)
    # cv2.imshow("Color Edge", dst)

    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()