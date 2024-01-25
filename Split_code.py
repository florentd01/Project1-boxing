#!/usr/bin/env python3

import os
import sys

from tqdm import tqdm

import cv2

# BASE_DIR = 'Videos/2'
SIDE_DIR = "side"
FRONT_DIR = "front"
multipliers = [1.5, 2, 4]


def create_directories(cur_dir):
    """
    Creates side and front dirs.
    """
    for dir_name in [SIDE_DIR, FRONT_DIR]:
        os.makedirs(os.path.join(cur_dir, dir_name), exist_ok=False)


def resize_video(video_path, multiplier):
    """
    Reduces the resolution of a video by a given multiplier and saves it.

    """
    # Verify if file exists
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Original dimensions
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # New dimensions
    new_width = int(orig_width / multiplier)
    new_height = int(orig_height / multiplier)

    # Create a video writer object
    dir_name, old_name = os.path.split(video_path)
    base, ext = os.path.splitext(old_name)
    new_video_path = os.path.join(dir_name, f"{multiplier}_{base}.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_video_path, fourcc, frame_rate, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Write the resized frame
        out.write(resized_frame)

    # Release everything
    cap.release()
    out.release()


def split_video(video_path):
    """
    Cuts video to left and right and saves it.
    """
    cur_dir = os.path.dirname(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Create data writer
    left_writer = cv2.VideoWriter(
        os.path.join(cur_dir, SIDE_DIR, os.path.basename(video_path)),
        cv2.VideoWriter_fourcc(*'mp4v'),
        frame_rate,
        (frame_width // 2, frame_height),
    )
    right_writer = cv2.VideoWriter(
        os.path.join(cur_dir, FRONT_DIR, os.path.basename(video_path)),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (frame_width // 2, frame_height),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # splitting videos and saving frames
        left_frame = frame[:, : frame_width // 2]
        right_frame = frame[:, frame_width // 2:]

        left_writer.write(left_frame)
        right_writer.write(right_frame)

    # clean resources
    cap.release()
    left_writer.release()
    right_writer.release()


def process_videos(cur_dir):
    """
    Processes videos in one dir.
    """
    # cut videos
    for filename in os.listdir(cur_dir):
        if filename.endswith(".mp4"):
            split_video(os.path.join(cur_dir, filename))

    # and then resize them
    side_dir = os.path.join(cur_dir, SIDE_DIR)
    front_dir = os.path.join(cur_dir, FRONT_DIR)
    for x_dir in [side_dir, front_dir]:
        for filename in os.listdir(x_dir):
            if filename.endswith(".mp4"):
                for multiplier in multipliers:
                    resize_video(os.path.join(x_dir, filename), multiplier)


if __name__ == "__main__":
    # for i in range(1,17):
        # BASE_DIR = f'Videos/{i}'
        BASE_DIR = 'Landmark_generator/test_video'
        for dirname in tqdm(os.listdir(BASE_DIR)):
            cur_dir = os.path.join(BASE_DIR, dirname)
            if os.path.isdir(cur_dir):
                try:
                    create_directories(cur_dir)
                    process_videos(cur_dir)
                except FileExistsError:
                    print(f'Directories `side` and `front` already exist in {cur_dir}, skipping for safety reasons.')