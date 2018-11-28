# OpenCV can open camera
import cv2 as cv

# Capture frame from a camera
cap = cv.VideoCapture(0)

while True:
    # Reads frame
    ret, frame = cap.read()

    # Display camera
    cv.imshow("Camera", frame)

    # Wait for Esc key to stop
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

# Close window
cap.release()

# De-allocate any associated memory usage
cv.destroyAllWindows()
