# gaze_detection_module.py
import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
import math
import os

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_LANDMARKS = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]
LEFT_IRIS_INDEX = 468
RIGHT_IRIS_INDEX = 473

def log_violation(reason):
    log = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), reason]], columns=["timestamp", "reason"])
    log.to_csv("violations.csv", mode='a', header=not os.path.exists("violations.csv"), index=False)

def get_iris_position_ratio(landmarks, eye_indices, iris_index):
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[1]]
    iris = landmarks[iris_index]
    eye_width = math.dist([left.x, left.y], [right.x, right.y])
    iris_dist = math.dist([left.x, left.y], [iris.x, iris.y])
    return iris_dist / eye_width

def process_frame(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(frame_rgb)
    mesh_results = face_mesh.process(frame_rgb)

    face_count = 0
    if detection_results.detections:
        face_count = len(detection_results.detections)
        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

    if face_count > 1:
        log_violation("Multiple faces detected")
        cv2.putText(frame, "Multiple Faces!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye_x = int(landmarks[33].x * w)
            right_eye_x = int(landmarks[263].x * w)
            nose_x = int(landmarks[1].x * w)
            eye_range = right_eye_x - left_eye_x
            gaze_center = (left_eye_x + right_eye_x) // 2

            if nose_x < gaze_center - eye_range * 0.2:
                log_violation("Eye Gaze: Looking LEFT")
                cv2.putText(frame, "Looking LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif nose_x > gaze_center + eye_range * 0.2:
                log_violation("Eye Gaze: Looking RIGHT")
                cv2.putText(frame, "Looking RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Looking CENTER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            left_ratio = get_iris_position_ratio(landmarks, LEFT_EYE_LANDMARKS, LEFT_IRIS_INDEX)
            right_ratio = get_iris_position_ratio(landmarks, RIGHT_EYE_LANDMARKS, RIGHT_IRIS_INDEX)
            gaze_ratio = (left_ratio + right_ratio) / 2

            if gaze_ratio < 0.35:
                log_violation("Iris Gaze: Looking LEFT")
                cv2.putText(frame, "Iris: Looking LEFT", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
            elif gaze_ratio > 0.65:
                log_violation("Iris Gaze: Looking RIGHT")
                cv2.putText(frame, "Iris: Looking RIGHT", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
            else:
                cv2.putText(frame, "Iris: Looking CENTER", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame
