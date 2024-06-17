import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import mido
import rtmidi
from model_class import HandGestureModel
from model_class2 import PosePanningModel

# Define Midi settings
MIDI = 'ON'  # Turn Midi Output 'ON' or 'OFF'
midi_channel = 1  # MIDI Output Channel
midi_control1 = 2  # First MIDI CC Message
midi_control2 = 3  # Second MIDI CC Message

# Define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Driver Bus 2')

# Initialize the models
hand_model = HandGestureModel()
hand_model.load_state_dict(torch.load('hand_gesture_model.pth'))
hand_model.eval()

pose_model = PosePanningModel()
pose_model.load_state_dict(torch.load('pose_panning_model.pth'))
pose_model.eval()

# Initialize MediaPipe Hands and Pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)
frame = cv2.flip(cap, 1)


def get_hand_landmarks_coordinates(hand_landmarks):
    coordinates = []
    for landmark in hand_landmarks.landmark:
        coordinates.extend([landmark.x, landmark.y, landmark.z])
    return coordinates

def get_pose_landmarks_coordinates(pose_landmarks):
    coordinates = []
    for landmark in pose_landmarks.landmark:
        coordinates.extend([landmark.x, landmark.y, landmark.z])
    return coordinates

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the hands and pose
    hand_results = hands.process(image)
    pose_results = pose.process(image)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark coordinates
            hand_landmarks = get_hand_landmarks_coordinates(hand_landmarks)

            # Convert the landmarks to a tensor
            hand_input_tensor = torch.tensor(hand_landmarks, dtype=torch.float32).unsqueeze(0)

            # Predict the hand gesture
            with torch.no_grad():
                hand_prediction = hand_model(hand_input_tensor)
                hand_gesture_value = hand_prediction.item()

            # Create and send midi cc message for hand gesture
            hand_cc = min(127, max(0, int(hand_gesture_value * 127)))
            if MIDI == 'ON':
                hand_msg = mido.Message('control_change', channel=midi_channel, control=midi_control1, value=hand_cc)
                port.send(hand_msg)

            # Display the predicted hand gesture value
            cv2.putText(frame, f"Hand Gesture Value: {hand_gesture_value:.2f} CC: {hand_cc}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the landmark coordinates
        pose_landmarks = get_pose_landmarks_coordinates(pose_results.pose_landmarks)

        # Convert the landmarks to a tensor
        pose_input_tensor = torch.tensor(pose_landmarks, dtype=torch.float32).unsqueeze(0)

        # Predict the panning position
        with torch.no_grad():
            pose_prediction = pose_model(pose_input_tensor)
            panning_value = pose_prediction.item()

        # Create and send midi cc message for panning position
        pose_cc1 = min(127, max(0, int(panning_value * 127)))
        pose_cc2 = 127 - pose_cc1
        if MIDI == 'ON':
            pose_msg1 = mido.Message('control_change', channel=midi_channel, control=midi_control2, value=pose_cc1)
            pose_msg2 = mido.Message('control_change', channel=midi_channel, control=midi_control2 + 1, value=pose_cc2)
            port.send(pose_msg1)
            port.send(pose_msg2)

        # Display the predicted panning value
        cv2.putText(frame, f"Panning Value: {panning_value:.2f} CC1: {pose_cc1} CC2: {pose_cc2}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the image
    cv2.putText(frame, f"Hand and Pose Detection. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Hand and Pose Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
