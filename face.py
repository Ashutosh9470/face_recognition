import cv2
import numpy as np

# Load pre-trained face detection model from OpenCV
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Load the model
face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104, 177, 123))
    print("Blob shape:", blob.shape)

    # Detect faces
    face_net.setInput(blob)
    detections = face_net.forward()
    print("Detections shape:", detections.shape)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Only consider detections above 50% confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            print(f"Detection {i}: Confidence: {confidence}, Box: {box}")

            # Draw a rectangle around detected face
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
