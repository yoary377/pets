import cv2
import numpy as np

cap = cv2.VideoCapture('vagon.mp4')

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# params for corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=1,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Couldn't read the first frame")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for orange color
    mask_orange = cv2.inRange(frame_hsv, lower_orange, upper_orange)

    # Apply the mask to the frame
    masked_frame_orange = cv2.bitwise_and(frame, frame, mask=mask_orange)

    # Find contours in the masked area
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_orange:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255),
                          2)  # Draw rectangles around detected orange areas

    cv2.imshow('Orange Color Tracking', frame)
    k = cv2.waitKey(25)
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
