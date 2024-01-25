import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import mediapipe as mp
import csv
import os
import pandas as pd
import pickle

def write_landmarks_to_list(landmarks, csv_data):
    # print(f"Landmark coordinates for frame {frame_number}:")
    frame_data= []

    for idx, landmark in enumerate(landmarks):
        # print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        frame_data.append(landmark.x)
        frame_data.append(landmark.y)
        frame_data.append(landmark.z)

    # print("\n")
    csv_data.append(frame_data)


def compute_media_pipe(video_path):
    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    landmark_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        result = pose.process(frame_rgb)

        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Add the landmark coordinates to the list
            write_landmarks_to_list(result.pose_landmarks.landmark, landmark_data)
    print(np.shape(landmark_data))

    # Release the VideoCapture object when done with each video
    cap.release()

    # landmark.apply

    return landmark_data

def compute_landmarks(participant, videos_folder_path, view, quality):
    # Iterate through each numbered folder (1-16)
    # num_participants = 16
    viable_moves = ['boxing stance', 'jab', 'cross', 'lead hook', 'rear hook', 'lead uppercut', 'rear uppercut']

    participant_data = []

    numbered_folder_path = os.path.join(videos_folder_path, str(participant))
    # Iterate through each action folder inside the numbered folder
    for action_folder in viable_moves:
        action_folder_path = os.path.join(numbered_folder_path, action_folder)
        view_folder_path = os.path.join(action_folder_path, view)

        # Iterate through each video file (1-5) inside the action folder
        for j in range(1, 6):

            # video_file_path = os.path.join(view_folder_path, f'export-{j}.mp4')

            # for different different qualities
            video_file_path = os.path.join(view_folder_path, f'{quality}export-{j}.mp4')

            print(video_file_path)

            participant_data.append(compute_media_pipe(video_file_path))

    return participant_data

def pad_data(data):
    max_size = max(len(lst) for lst in data)
    # print(max_size)
    padded_data = data
    for i in range(len(data)):
        for j in range(max_size - len(data[i])):
            padding_list = [-10] * len(data[i][0])
            padded_data = data[i].append(padding_list)
    
    return padded_data

def save_data(participant, final_data, view, quality):
    with open(f'Landmarks/{view}_{quality}/{participant}{view}_{quality}landmarks.pkl', 'wb') as file:
        pickle.dump(final_data, file)


def main(participant, video_folder_path, view, quality):
    landmarks = compute_landmarks(participant, video_folder_path, view, quality)
    # final_data = pad_data(landmarks)
    # save_data(participant, final_data, view, quality)
    save_data(participant, landmarks, view, quality)
    print("DONE")


################################################################
videos_folder_path = 'Videos'
view = 'front' # side
quality = '4_'
# quality = '1.5_' # need _

participant = 16

main(participant, videos_folder_path, view, quality)