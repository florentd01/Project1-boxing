from pose_model import PoseModel
from logger_setup import setup_logger
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2

logger = setup_logger()
config = {
    'model_path': r'__pycache__\pose_model.cpython-311.pyc',
    'threshold': 0.7,
    'colors': {'point': (255, 0, 0), 'line': (0, 255, 0)}
}


def process_video(filename, config):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Failed to open video file at {filename}")
        return
    pose_model = PoseModel()  # Define the pose_model variable
    # Define the landmark_names dictionary
    landmark_names = {
        0: "nose",
        1: "left eye (inner)",
        2: "left eye",
        3: "left eye (outer)",
        4: "right eye (inner)",
        5: "right eye",
        6: "right eye (outer)",
        7: "left ear",
        8: "right ear",
        9: "mouth (left)",
        10: "mouth (right)",
        11: "left shoulder ",
        12: "right shoulder",
        13: "left elbow ",
        14: "right elbow",
        15: "left wrist",
        16: "right wrist",
        17: "left pinky",
        18: "right pinky",
        19: "left index",
        20: "right index",
        21: "left thumb",
        22: "right thumb",
        23: "left hip",
        24: "right hip",
        25: "left knee",
        26: "right knee",
        27: "left ankle",
        28: "right ankle",
        29: "left heel",
        30: "right heel",
        31: "left foot index",
        32: "right foot index"

    }
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, landmarks = pose_model.process_image(frame)
        if landmarks is not None:
            print(landmarks[15])
            for landmark in landmarks:
                #print(landmark)
                logger.info(f"Landmark  : {landmark}")


        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Create a simple GUI for file selection
Tk().withdraw()
filename = askopenfilename()

# Call the function with the selected file and a configuration
process_video(filename, config)
