import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam capture
cap = cv2.VideoCapture(0)

# Finger tip landmark indices (Mediapipe index for fingertips)
finger_tips = [4, 8, 12, 16, 20]  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB (Mediapipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    finger_count = 0  # Reset finger count

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark positions
            landmarks = hand_landmarks.landmark

            # Count fingers (Compare finger tip y-position with the lower joint)
            if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:  # Thumb (left vs. right hand)
                finger_count += 1

            for i in range(1, 5):  # Index to pinky fingers
                if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y:
                    finger_count += 1

    # Display the count on the screen
    cv2.putText(frame, f'Fingers: {finger_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger Count", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
