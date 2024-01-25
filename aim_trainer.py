"""
This script is a simple example of a possible boxing aim assistance trainer.
From either a video or webcam input, the script uses mediapipe to extract key-points.

The script displays the video feed and superimposes some orange circles on top that serve
as targets for the user. 

The script uses the extracted keypoint data to detect when a punch
is thrown and whether the punch was within the target. If a hit is registered, the script
moves on to a new target.

Target locations can be specified by modifying the circle_coords list in the main function.
Each tuple represents the (x,y) coordinates of the center of each target on the image.

"""

import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from pose_model import PoseModel

import math


def get_file():
    """
    Uses the tkinter library to request a file

    returns a file path
    """
    Tk().withdraw()
    return askopenfilename()


def check_tolerance(center, wrist, frame_dims):
    """
    Checks if the wrist is within the target
    """
    wrist_abs = (int(wrist[0] * frame_dims[0]), int(wrist[1] * frame_dims[1]))
    center_norm = (center[0] / frame_dims[0], center[1] / frame_dims[1])
    return math.dist(center_norm, wrist) < 0.1


def run(wrists, circles, webcam_bool):
    if webcam_bool:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"Failed to open webcam")
            return
    else:
        filename = get_file()
        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            print(f"Failed to open video file at {filename}")
            return

    pose_model = PoseModel()

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, landmarks = pose_model.process_image(frame)
        frame2 = cv2.circle(frame, circles[counter], 20, (0, 150, 255), 2)

        # print(frame.shape)
        if landmarks is not None:
            if landmarks[15].z < -1:
                wrist_coords = (landmarks[15].x, landmarks[15].y)
                hit = check_tolerance(circles[counter], wrist_coords, (width, height))
                if hit:
                    print("HIT")
                    counter += 1

            pass

        cv2.imshow('Aim Trainer', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    wrists_indices = [15, 16]
    circle_coords = [(255, 300), (150, 150)]


    #run(wrists_indices, circle_coords, True)
    run(wrists_indices, circle_coords, False)



if __name__ == "__main__":
    main()
