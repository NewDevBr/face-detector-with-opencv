import cv2

# Loading classifier
xml_haar_cascade = 'haarcascade_frontalface_alt2.xml'
face_classifier = cv2.CascadeClassifier(xml_haar_cascade)

# Initiating camera
capture = cv2.VideoCapture(0)

# Setting camera configs
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

# Subscription on video
while True:
    # Reading camera image
    ret, frame = capture.read()
    # Converting frame to gray
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Applying faces detector classifier
    faces = face_classifier.detectMultiScale(frame_gray)

    # Drawing a brown rectangle in faces
    for x, y, z, h in faces:
        cv2.rectangle(frame, (x, y), (x + z, y + h), (38, 48, 86), 3)

    # Showing a dialog box with rectangle
    cv2.imshow("webcam", frame)

    # Stop this app if users press key q (quit)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
