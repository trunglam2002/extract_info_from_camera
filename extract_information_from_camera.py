import cv2
from deepface import DeepFace
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define backends
backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'fastmtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]
backend = backends[7]

# Connect to MongoDB Atlas
uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]  # MongoDB database
collection = db["user_info"]  # MongoDB collection

# Draw bounding box and display user information


def draw_bounding_box(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Analyze emotion and match with database


def analyze_face(face_image, facial_area, distance_threshold, db_path):
    emotion_result = DeepFace.analyze(img_path=face_image, actions=[
                                      'emotion'], enforce_detection=False, detector_backend=backend)

    if emotion_result is not None and len(emotion_result) > 0:
        dominant_emotion = emotion_result[0]['dominant_emotion']
        results = DeepFace.find(img_path=face_image, db_path=db_path,
                                enforce_detection=False, detector_backend=backend, silent=True)

        return (facial_area, dominant_emotion, results)
    return (facial_area, None, None)


# Using the camera
cap = cv2.VideoCapture(0)

# Distance threshold for identity matching
distance_threshold = 0.6
db_path = "test/my_db"
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 512x512
    frame = cv2.resize(frame, (512, 512))

    frame_counter += 1
    if frame_counter % 3 != 0:  # Process every third frame
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_faces = DeepFace.extract_faces(
        img_rgb, enforce_detection=False, detector_backend=backend)

    if detected_faces is not None and len(detected_faces) > 0:
        with ThreadPoolExecutor() as executor:
            futures = []
            for detected_face in detected_faces:
                facial_area = detected_face['facial_area']
                face_image = detected_face['face']
                futures.append(executor.submit(
                    analyze_face, face_image, facial_area, distance_threshold, db_path))

            for future in as_completed(futures):
                facial_area, dominant_emotion, results = future.result()
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                # Process results
                if dominant_emotion is not None:
                    if results is not None and len(results) > 0:
                        combined_results = pd.concat(
                            results, ignore_index=True)
                        if not combined_results.empty:
                            best_match_index = combined_results['distance'].idxmin(
                            )
                            best_result = combined_results.iloc[best_match_index]
                            identity_path = best_result['identity'].replace(
                                "\\", "/")
                            distance = best_result['distance']

                            if distance < distance_threshold:
                                name = identity_path.split("/")[-2]
                                user_info = collection.find_one({"name": name})
                                if user_info:
                                    user_text = f"{user_info['name']} - Emotion: {dominant_emotion}, phone: {user_info['phone']}, account_number: {user_info['account_number']}"
                                    draw_bounding_box(
                                        frame, x, y, w, h, f"{name} - {dominant_emotion}")
                                else:
                                    draw_bounding_box(
                                        frame, x, y, w, h, f"{name} - {dominant_emotion}")
                            else:
                                draw_bounding_box(
                                    frame, x, y, w, h, f"Unknown - {dominant_emotion}")
                    else:
                        draw_bounding_box(frame, x, y, w, h,
                                          f"Unknown - {dominant_emotion}")

    cv2.imshow('Camera Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
