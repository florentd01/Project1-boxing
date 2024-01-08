import cv2
import mediapipe as mp


class PoseModel:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.pose = mp.solutions.pose.Pose()

    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        landmarks = None  # Define landmarks before the if block
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), landmarks
