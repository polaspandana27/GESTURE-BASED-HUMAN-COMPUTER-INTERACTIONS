import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Streamlit UI setup
st.title("Gesture-based Human-Computer Interaction")
st.write("Show your hand gestures to interact!")

# Capture webcam input using Streamlit
frame_input = st.camera_input("Enable your webcam and show hand gestures:")

if frame_input is not None:
    # Convert the captured frame to an OpenCV image
    bytes_data = frame_input.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)  # Flip horizontally for mirror view

    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe Hands
    results = hands.process(rgb_frame)

    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [lm for lm in hand_landmarks.landmark]
            finger_fold_status = []

            for tip in finger_tips:
                if lm_list[tip].x < lm_list[tip - 3].x:
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            gestures = []

            # Gesture conditions
            if all(finger_fold_status):
                gestures.append("LIKE 👍")

            if lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y and all(finger_fold_status):
                gestures.append("DISLIKE 👎")

            thumb_x, thumb_y = lm_list[thumb_tip].x, lm_list[thumb_tip].y
            index_x, index_y = lm_list[8].x, lm_list[8].y
            if np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2) < 0.05 and not any(finger_fold_status[1:]):
                gestures.append("OK 👌")

            if np.sqrt((lm_list[4].x - lm_list[16].x)**2 + (lm_list[4].y - lm_list[16].y)**2) < 0.05 and \
               np.sqrt((lm_list[16].x - lm_list[20].x)**2 + (lm_list[16].y - lm_list[20].y)**2) < 0.05 and \
               not finger_fold_status[0] and not finger_fold_status[1]:
                gestures.append("PEACE ✌️")

            if finger_fold_status[0] and finger_fold_status[1] and finger_fold_status[3] and \
               np.sqrt((lm_list[4].x - lm_list[20].x)**2 + (lm_list[4].y - lm_list[20].y)**2) > 0.4:
                gestures.append("CALL ME 🤙")

            if all([lm_list[i].y < lm_list[i - 1].y for i in finger_tips + [thumb_tip]]):
                gestures.append("STOP ✋")

            if lm_list[8].y < lm_list[6].y and all([lm_list[i].y > lm_list[i - 1].y for i in [12, 16, 20]]) and lm_list[4].x > lm_list[3].x:
                gestures.append("FORWARD 👆")

            if lm_list[4].y < lm_list[2].y and lm_list[8].x < lm_list[6].x and all([lm_list[i].x > lm_list[i - 1].x for i in [12, 16, 20]]) and lm_list[5].x < lm_list[0].x:
                gestures.append("LEFT 👈")

            if lm_list[4].y < lm_list[2].y and lm_list[8].x > lm_list[6].x and all([lm_list[i].x < lm_list[i - 1].x for i in [12, 16, 20]]):
                gestures.append("RIGHT 👉")

            if lm_list[8].y < lm_list[6].y and lm_list[20].y < lm_list[19].y and not any([finger_fold_status[1], finger_fold_status[2]]):
                gestures.append("I LOVE YOU 🤟")

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        st.image(frame, channels="BGR")
        st.write("**Detected Gestures:**", ", ".join(gestures))