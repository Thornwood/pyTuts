# OpenCV can open camera
import cv2 as cv

# Capture frame from a camera
cap = cv.VideoCapture(0)

filterTuple = ("cv.COLOR_BGR2RGBA", "cv.COLOR_BGR2GRAY")
filterke = 0

while True:
    # Reads frame
    ret, frame = cap.read()
    filteredPicture = frame

    # Wait for Esc key to stop
    k = cv.waitKey(5)

    if k == ord('a'):
        filteredPicture = frame
    else:
        filteredPicture = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if k == 27:
        break

    # Display camera
    cv.imshow("Camera", filteredPicture)

# Close window
cap.release()

# De-allocate any associated memory usage
cv.destroyAllWindows()
