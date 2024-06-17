import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os

# Directory to save the data to
path = 'Smart-Media-Technology-main/hand_data/'

# Global variable to set record time for each gesture in seconds
RECORD_TIME = 10

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

# Function to capture landmarks for a set duration
def record_landmarks(duration, gesture_label):
    start_time = time.time()
    landmarks_list = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the hands.
        results = hands.process(image)

        # Draw hand landmarks.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
                landmarks_list.append((landmarks, gesture_label))

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display the image with instructions.
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording {gesture_label}... {duration - int(elapsed_time)}s left. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Hand Gesture Calibration', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    return landmarks_list

# Data collection with automated recording
def calibrate_gesture(gesture_index, record_time):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture_name = ["fully closed",
                        "25% open",
                        "50% open",
                        "75% open",
                        "100% open"]

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display instructions
        cv2.putText(frame, f"Show {gesture_name[gesture_index]} hand and press 'r' to record. Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Hand Gesture recording', frame)

        if cv2.waitKey(5) & 0xFF == ord('r'):
            landmarks = record_landmarks(record_time, gesture_index / 4)
            cv2.destroyAllWindows()  # Close the current window before opening the next one
            return landmarks

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Recording samples
full_closed_samples = calibrate_gesture(0, RECORD_TIME)
quarter_closed_samples = calibrate_gesture(1, RECORD_TIME)
half_closed_samples = calibrate_gesture(2, RECORD_TIME)
three_quarter_closed_samples = calibrate_gesture(3, RECORD_TIME)
open_palm_samples = calibrate_gesture(4, RECORD_TIME)

# Combine all samples for calibration and training
all_samples = full_closed_samples + quarter_closed_samples + half_closed_samples + three_quarter_closed_samples + open_palm_samples

# Better way to name files - avoids merge conflicts
# (unless multiple people happen to do training at the exact same second)
date_time = time.asctime().replace(" ", "_")

# Save gesture data to CSV
with open(path + f'hand_gesture_data_{date_time}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gesture'] + [f'[{i}]' for i in range(21)])
    for landmarks, gesture_label in all_samples:
        row = [gesture_label] + [f'{x},{y},{z}' for (x, y, z) in landmarks]
        writer.writerow(row)

print("Calibration data saved successfully!")

# Keep showing the screen
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the hands.
    results = hands.process(image)

    # Draw hand landmarks.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Data saved successfully! Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
