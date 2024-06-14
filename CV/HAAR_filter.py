import cv2

face_cascade = cv2.CascadeClassifier("haar_filters/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haar_filters/haarcascade_eye.xml")
cat_cascade = cv2.CascadeClassifier("haar_filters/haarcascade_frontalcatface.xml")
input_video_path = 'me2.mp4'  # Update with your video file
cap = cv2.VideoCapture(input_video_path)
# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
# Process and display each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the frame
    resized_frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.5)
    # resized_frame = cv2.resize(frame, (width, height))
    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h),
                      (255, 0, 0), 2)
    cv2.putText(resized_frame, 'Face', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = resized_frame[y:y + h, x:x + w]
    # Perform eye detection within the face ROI
    eyes = eye_cascade.detectMultiScale(roi_gray,
                                        scaleFactor=1.3, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey +
                                            eh), (0, 255, 0), 2)
    cv2.putText(roi_color, 'Eyes', (ex, ey - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cat = cat_cascade.detectMultiScale(roi_color)
    for (wx, wy, ww, wh) in cat:
        cv2.rectangle(roi_color, (wx, wy), (wx + ww, wy +
                                            wh), (0, 0, 255), 2)
    cv2.putText(roi_color, 'Cat face', (wx, wy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # Display the resized frame
    cv2.imshow('Resized Video', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()
