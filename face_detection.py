import cv2
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor
from face_recognition import process_face_recognition
from emotion_detection import process_emotion_detection
from anti_spoofing import process_anti_spoofing
from database import query_user_info
from utils import draw_bounding_box, preprocess_coordinate
import os
import logging

detector = MTCNN()


def process_face_detection(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)
    return results


def process_frame(frame, frame_count, skip_frames=5, previous_faces=None):
    if previous_faces is None:
        previous_faces = []

    if frame_count % skip_frames != 0:
        for (x, y, w, h, name, emotion, is_real) in previous_faces:
            draw_bounding_box(frame, x, y, w, h, name, emotion, is_real)
        return frame, previous_faces

    results = process_face_detection(frame)
    current_faces = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        face_boxes = []

        for result in results:
            x, y, w, h = result['box']
            face_image = frame[y:y + h, x:x + w]
            sequence_coordinate = preprocess_coordinate(x, y, w, h)
            futures.append(executor.submit(
                process_face_recognition, face_image))
            futures.append(executor.submit(
                process_emotion_detection, frame, sequence_coordinate))
            futures.append(executor.submit(process_anti_spoofing, face_image))
            face_boxes.append((x, y, w, h))

        for i in range(0, len(futures), 3):
            name = futures[i].result()
            emotion = futures[i + 1].result()
            is_real = futures[i + 2].result()
            user_info = query_user_info(name)
            if user_info:
                logging.info(f"User Info: {user_info}")
            x, y, w, h = face_boxes[i // 3]
            draw_bounding_box(frame, x, y, w, h, name, emotion, is_real)
            current_faces.append((x, y, w, h, name, emotion, is_real))

    return frame, current_faces
