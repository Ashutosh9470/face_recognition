import cv2
import numpy as np

# Load pre-trained deep learning model
AGE_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
AGE_MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/face_detection_yunet/face-detection-adas-0001.caffemodel"

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# Age labels based on training data
AGE_RANGES = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to blob (for deep learning model)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(227, 227),
                                 mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    print("Blob shape:", blob.shape)

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    print("Age predictions:", age_preds)
    age = AGE_RANGES[age_preds[0].argmax()]  # Get predicted age range
    print("Predicted age range:", age)

    # Display predicted age on the screen
    cv2.putText(frame, f"Age: {age}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Age Prediction", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
