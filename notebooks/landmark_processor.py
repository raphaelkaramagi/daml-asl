# landmark_processor.py - Worker function for parallel processing
# This file exists because Jupyter notebooks can't pickle functions for multiprocessing

import os
# Disable GPU for MediaPipe - required for parallel processing on macOS
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

def process_single_image(file_path):
    """Process one image and return hand landmarks"""
    import mediapipe as mp
    import cv2
    import numpy as np

    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    try:
        image = cv2.imread(file_path)
        if image is None:
            return [np.nan] * 63

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_model.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist_coords = hand_landmarks.landmark[0]

            landmark_row = []
            for landmark in hand_landmarks.landmark:
                relative_x = landmark.x - wrist_coords.x
                relative_y = landmark.y - wrist_coords.y
                relative_z = landmark.z - wrist_coords.z
                landmark_row.extend([relative_x, relative_y, relative_z])
            return landmark_row
        else:
            return [np.nan] * 63
    except Exception as e:
        # Return NaN if any error occurs
        return [np.nan] * 63
    finally:
        # Clean up MediaPipe resources
        hands_model.close()
