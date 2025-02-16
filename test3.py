import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Webcam setup
cap = cv2.VideoCapture(0)

# Smoothing variables
prev_x, prev_y = 0, 0
smoothing_factor = 0.2  # Adjust for better smoothness

# Click control variables
click_threshold = 40  # Adjust for pinch sensitivity
last_click_time = 0
click_delay = 0.3  # Prevents multiple unwanted clicks

# Scroll control variables
scroll_sensitivity = 30  # Adjust for scroll speed

# Drag & Drop variables
dragging = False
drag_start_time = 0
drag_hold_time = 1.0  # Hold pinch for 1 second to start dragging

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    frame_height, frame_width, _ = frame.shape

    # Convert BGR to RGB (Mediapipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for index finger tip (8), middle finger tip (12), and thumb tip (4)
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]

            # Convert hand coordinates to screen coordinates
            cursor_x = int(index_finger.x * screen_width)
            cursor_y = int(index_finger.y * screen_height)

            # Apply smoothing for cursor movement
            smoothed_x = int(prev_x + (cursor_x - prev_x) * smoothing_factor)
            smoothed_y = int(prev_y + (cursor_y - prev_y) * smoothing_factor)
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y  # Update previous position

            # Euclidean distance for pinch detection (Left Click / Drag & Drop)
            thumb_x, thumb_y = int(thumb.x * frame_width), int(thumb.y * frame_height)
            index_x, index_y = int(index_finger.x * frame_width), int(index_finger.y * frame_height)
            distance = np.linalg.norm([index_x - thumb_x, index_y - thumb_y])

            current_time = time.time()

            # Start Dragging if pinch is held for 1 sec
            if distance < click_threshold:
                if not dragging:
                    if drag_start_time == 0:
                        drag_start_time = current_time
                    elif current_time - drag_start_time >= drag_hold_time:
                        pyautogui.mouseDown()
                        dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()  # Release the item
                    dragging = False
                drag_start_time = 0  # Reset drag timer

            # Detect scrolling motion
            index_y_screen = int(index_finger.y * screen_height)
            middle_y_screen = int(middle_finger.y * screen_height)

            # If both index and middle fingers are up, detect vertical movement for scrolling
            if index_finger.y < hand_landmarks.landmark[6].y and middle_finger.y < hand_landmarks.landmark[10].y:
                scroll_amount = (middle_y_screen - index_y_screen) / scroll_sensitivity
                
                if scroll_amount > 1:  # Scroll Down
                    pyautogui.scroll(-3)
                elif scroll_amount < -1:  # Scroll Up
                    pyautogui.scroll(3)

    # Display the webcam feed
    cv2.imshow("Virtual Mouse with Drag & Drop", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
