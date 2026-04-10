import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# UI
st.title("🖐️ Sign Language Translator")

start = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = None

predictions_buffer = []

# =========================
# Run only if checkbox is ON
# =========================
if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not accessible ❌")
    else:
        st.success("Camera started ✅")

        while start:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read camera")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    landmarks = np.array(landmarks).reshape(1, -1)

                    # Predict
                    prediction = model.predict(landmarks)[0]

                    predictions_buffer.append(prediction)

                    if len(predictions_buffer) > 10:
                        predictions_buffer.pop(0)

                    final_prediction = max(
                        set(predictions_buffer),
                        key=predictions_buffer.count
                    )

                    cv2.putText(
                        frame,
                        f"Prediction: {final_prediction}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()

else:
    st.info("Click 'Start Camera' to begin")