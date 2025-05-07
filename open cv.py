import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Volume control variables
vol_min, vol_max = 0, 100
current_vol = 50
last_vol_change_time = 0
vol_change_duration = 0.3  # seconds between volume steps
vol_bar_height = 300
vol_bar_width = 40

# Brightness control variables
brightness = 50
brightness_bar_height = 300
brightness_bar_width = 40

# Smoothing variables
prev_distance = 0
smooth_factor = 0.5

# Animation variables
vol_change_indicator = ""
vol_indicator_start_time = 0
vol_indicator_duration = 1  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time at the start of each frame
    current_time = time.time()

    # Flip the frame horizontally for a better experience
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get key landmarks
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            # Convert coordinates to pixel values
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            middle_mcp_x, middle_mcp_y = int(middle_mcp.x * w), int(middle_mcp.y * h)

            # Calculate distance between thumb and index finger (smoothed)
            current_distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
            smoothed_distance = prev_distance * smooth_factor + current_distance * (1 - smooth_factor)
            prev_distance = smoothed_distance

            # Determine hand orientation
            dx = middle_mcp_x - wrist_x
            dy = middle_mcp_y - wrist_y
            angle = math.degrees(math.atan2(dy, dx))

            # Vertical hand (angle between 45 and 135 degrees or -45 and -135 degrees)
            is_vertical = (45 < abs(angle) < 135)
            mode = "volume" if is_vertical else "brightness"

            # Draw landmarks and connections
            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

            # Draw circles at thumb and index finger tips
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)

            # Control based on mode
            if mode == "volume":
                # Calculate volume percentage
                vol_percentage = np.interp(smoothed_distance, [50, 200], [vol_min, vol_max])

                # Only change volume if enough time has passed
                if current_time - last_vol_change_time > vol_change_duration:
                    if vol_percentage > current_vol + 5:  # Increasing volume
                        pyautogui.press("volumeup")
                        current_vol = min(vol_max, current_vol + 5)
                        vol_change_indicator = "Volume +"
                        vol_indicator_start_time = current_time
                        last_vol_change_time = current_time
                    elif vol_percentage < current_vol - 5:  # Decreasing volume
                        pyautogui.press("volumedown")
                        current_vol = max(vol_min, current_vol - 5)
                        vol_change_indicator = "Volume -"
                        vol_indicator_start_time = current_time
                        last_vol_change_time = current_time

                # Draw volume bar
                vol_bar_y = int(np.interp(current_vol, [vol_min, vol_max], [h - 50, h - 50 - vol_bar_height]))
                cv2.rectangle(frame, (50, h - 50), (50 + vol_bar_width, h - 50 - vol_bar_height), (0, 255, 0), 2)
                cv2.rectangle(frame, (50, h - 50), (50 + vol_bar_width, vol_bar_y), (0, 255, 0), -1)
                cv2.putText(frame, f"{current_vol}%", (50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Visual feedback
                cv2.putText(frame, f"Volume Mode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:  # brightness mode
                # Calculate brightness percentage
                brightness = int(np.interp(smoothed_distance, [50, 200], [0, 100]))

                # Draw brightness bar
                bright_bar_y = int(np.interp(brightness, [0, 100], [h - 50, h - 50 - brightness_bar_height]))
                cv2.rectangle(frame, (w - 100, h - 50),
                              (w - 100 + brightness_bar_width, h - 50 - brightness_bar_height), (255, 255, 0), 2)
                cv2.rectangle(frame, (w - 100, h - 50), (w - 100 + brightness_bar_width, bright_bar_y), (255, 255, 0),
                              -1)
                cv2.putText(frame, f"{brightness}%", (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Visual feedback
                cv2.putText(frame, f"Brightness Mode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Display distance
            cv2.putText(frame, f"Distance: {int(smoothed_distance)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show volume change indicator at bottom center
    if current_time - vol_indicator_start_time < vol_indicator_duration:
        text_size = cv2.getTextSize(vol_change_indicator, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 30  # Position at bottom of screen
        cv2.putText(frame, vol_change_indicator, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)



    # Display the frame
    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()