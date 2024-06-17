import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os

# Directory to save the data to
path = 'Smart-Media-Technology-main/pose_data/'

# Global variable to set record time for each gesture in seconds
RECORD_TIME = 10

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

# Function to capture landmarks for a set duration
def record_landmarks(duration, pose_label):
    start_time = time.time()
    landmarks_list = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose.
        results = pose.process(image)

        # Draw pose landmarks.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark]
            landmarks_list.append((landmarks, pose_label))

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display the image with instructions.
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording {pose_label}... {duration - int(elapsed_time)}s left. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Pose Calibration', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    return landmarks_list

# Data collection with automated recording
def calibrate_pose(pose_index, record_time):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_name = ["left", "center", "right"]

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display instructions
        cv2.putText(frame, f"Show pose for {pose_name[pose_index]} and press 'r' to record. Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Pose Recording', frame)

        if cv2.waitKey(5) & 0xFF == ord('r'):
            landmarks = record_landmarks(record_time, pose_name[pose_index])
            cv2.destroyAllWindows()  # Close the current window before opening the next one
            return landmarks

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Recording samples for the three poses
left_samples = calibrate_pose(0, RECORD_TIME)
center_samples = calibrate_pose(1, RECORD_TIME)
right_samples = calibrate_pose(2, RECORD_TIME)

# Combine all samples for calibration and training
all_samples = left_samples + center_samples + right_samples

# Better way to name files - avoids merge conflicts
# (unless multiple people happen to do training at the exact same second)
date_time = time.asctime().replace(" ", "_")

# Save pose data to CSV
with open(path + f'pose_data_{date_time}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['position'] + [f'[{i}]' for i in range(33)])  # 33 landmarks for pose model
    for landmarks, pose_label in all_samples:
        row = [pose_label] + [f'{x},{y},{z}' for (x, y, z) in landmarks]
        writer.writerow(row)

print("Calibration data saved successfully!")

# Keep showing the screen
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose.
    results = pose.process(image)

    # Draw pose landmarks.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Data saved successfully! Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Pose Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
