import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

st.title("Gesture-based Human-Computer Interaction")
st.write("Show your hand gestures to interact!")

frame_input = st.camera_input("Enable your webcam and show hand gestures:")

if frame_input is not None:
    bytes_data = frame_input.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            finger_fold = [lm[tip].y > lm[tip - 2].y for tip in finger_tips]

            gestures = []

            # LIKE: All fingers folded except thumb extended
            if finger_fold == [True, True, True, True] and lm[thumb_tip].x < lm[thumb_tip - 1].x:
                gestures.append("LIKE 👍")

            # DISLIKE: All fingers folded, thumb extended downward
            if finger_fold == [True, True, True, True] and lm[thumb_tip].x > lm[thumb_tip - 1].x:
                gestures.append("DISLIKE 👎")

            # OK: Thumb and index form a circle, other fingers extended
            if np.linalg.norm(np.array([lm[thumb_tip].x, lm[thumb_tip].y]) - np.array([lm[8].x, lm[8].y])) < 0.03 and all(not f for f in finger_fold[1:]):
                gestures.append("OK 👌")

            # PEACE: Index and middle extended, others folded
            if finger_fold == [False, False, True, True]:
                gestures.append("PEACE ✌️")

            # CALL ME: Thumb and pinky extended, others folded
            if finger_fold == [True, True, True, False] and not finger_fold[0]:
                gestures.append("CALL ME 🤙")

            # STOP: All fingers extended
            if finger_fold == [False, False, False, False]:
                gestures.append("STOP ✋")

            # FORWARD: Index extended, others folded
            if finger_fold == [False, True, True, True]:
                gestures.append("FORWARD 👆")

            # LEFT: Thumb extended left, others folded
            if finger_fold == [True, True, True, True] and lm[thumb_tip].x < lm[thumb_tip - 1].x:
                gestures.append("LEFT 👈")

            # RIGHT: Thumb extended right, others folded
            if finger_fold == [True, True, True, True] and lm[thumb_tip].x > lm[thumb_tip - 1].x:
                gestures.append("RIGHT 👉")

            # I LOVE YOU: Thumb, index, pinky extended, others folded
            if finger_fold == [False, True, True, False] and not finger_fold[0]:
                gestures.append("I LOVE YOU 🤟")

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        st.image(frame, channels="BGR")
        st.write("**Detected Gestures:**", ", ".join(gestures))