import math

import cv2
import numpy as np


class PupilDetection:
    def __init__(self, video_path):
        self._cap = cv2.VideoCapture(video_path)

    def filter_contours(self, contours):
        """
        Filter contours based on area, symmetry, and circularity conditions.
        """
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            rect = cv2.boundingRect(contour)
            x, y, width, height = rect
            radius = 0.25 * (width + height)

            # Area condition to filter small and large contours
            area_condition = (70 <= area <= 200)

            # Symmetry condition to filter contours with similar width and height
            symmetry_condition = (abs(1 - float(width) / float(height)) <= 0.2)

            # Circularity condition to filter more circular contours
            fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.3)

            if area_condition and symmetry_condition and fill_condition:
                filtered_contours.append(contour)

        return filtered_contours

    def detect_pupil(self, frame):
        """
        Detect and mark the pupil with a dot in the given frame.
        """
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use HoughCircles to detect circles in the image
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=150,
            param1=50,
            param2=30,
            minRadius=75,
            maxRadius=100
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]

                # Mark the center of the detected circle as the pupil
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow("Pupil Detection", frame)

    def start_detection(self):
        """
        Start the pupil detection process for each frame in the video.
        """
        while True:
            # Read a frame from the video
            ret, frame = self._cap.read()

            # Break the loop if there are no more frames
            if not ret:
                break

            # Detect pupil in the current frame
            self.detect_pupil(frame)

            # Display the frame
            cv2.imshow("Pupil Detection", frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # Release the video capture object and close the display window
        self._cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "eye5.mp4"  # Replace with your video path
    detector = PupilDetection(video_path)
    detector.start_detection()
