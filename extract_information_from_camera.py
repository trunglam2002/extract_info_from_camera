import cv2
from deepface import DeepFace
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# Define backends
backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface'
]
backend = backends[7]  # Use YOLOv8 for face detection

# Connect to MongoDB Atlas
uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]
collection = db["user_info"]

# Function to draw bounding boxes and display user information


def draw_bounding_box(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Function to analyze face: emotion detection + matching with database


def analyze_face(face_image, facial_area, distance_threshold, db_path):
    # Analyze emotion
    emotion_result = DeepFace.analyze(
        img_path=face_image, actions=['emotion'],
        enforce_detection=False, detector_backend=backend
    )
    dominant_emotion = emotion_result[0]['dominant_emotion'] if emotion_result else 'Unknown'

    # Find face in the database
    results = DeepFace.find(
        img_path=face_image, db_path=db_path,
        enforce_detection=False, detector_backend=backend, silent=True
    )

    return (facial_area, dominant_emotion, results)


# Initialize queues
frame_queue = queue.Queue()
face_queue = queue.Queue()
result_queue = queue.Queue()

# Function to read frames from the camera


def camera_reader():
    cap = cv2.VideoCapture(0)

    width = 640
    height = 480
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frame_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

# Function to extract faces from every nth frame


def face_extractor():
    frame_counter = 0
    while True:
        frame = frame_queue.get()
        frame_counter += 1
        # Process every 10th frame
        if frame_counter % 10 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = DeepFace.extract_faces(
                img_rgb, enforce_detection=False, detector_backend=backend)
            if detected_faces:
                for detected_face in detected_faces:
                    facial_area = detected_face['facial_area']
                    face_image = detected_face['face']
                    face_queue.put((face_image, facial_area, frame))

# Function to analyze faces and compare with the database using ThreadPoolExecutor


def face_analyzer():
    with ThreadPoolExecutor() as executor:
        while True:
            face_data = face_queue.get()
            if face_data:
                face_image, facial_area, frame = face_data
                futures = executor.submit(
                    analyze_face, face_image, facial_area, 0.6, "test/my_db")
                facial_area, dominant_emotion, results = futures.result()
                result_queue.put(
                    (facial_area, dominant_emotion, results, frame))

# Function to lookup MongoDB and display result


def mongodb_lookup():
    while True:
        facial_area, dominant_emotion, results, frame = result_queue.get()

        # Initialize bounding box variables
        x, y, w, h = 0, 0, 0, 0  # Default values for no face detected

        # Only set coordinates if facial_area is not None
        if facial_area is not None:
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

        # Process results
        if results is not None and len(results) > 0:
            combined_results = pd.concat(results, ignore_index=True)
            if not combined_results.empty:
                best_match_index = combined_results['distance'].idxmin()
                best_result = combined_results.iloc[best_match_index]
                identity_path = best_result['identity'].replace("\\", "/")
                distance = best_result['distance']

                if distance < 0.6:
                    # Get name from folder path
                    name = identity_path.split("/")[-2]
                    # Find user info from MongoDB
                    user_info = collection.find_one({"name": name})
                    if user_info:
                        print(f"User: {user_info['name']}, Emotion: {dominant_emotion}, "
                              f"Phone: {user_info['phone']}, Account Number: {user_info['account_number']}")
                        draw_bounding_box(frame, x, y, w, h, user_info['name'])
                    else:
                        draw_bounding_box(frame, x, y, w, h, name)
                else:
                    draw_bounding_box(frame, x, y, w, h, "Unknown")
            else:
                draw_bounding_box(frame, x, y, w, h, "Unknown")
        else:
            draw_bounding_box(frame, x, y, w, h, "Unknown")

        # Display the result
        cv2.imshow('Camera Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Start threads
camera_thread = threading.Thread(target=camera_reader)
extractor_thread = threading.Thread(target=face_extractor)
analyzer_thread = threading.Thread(target=face_analyzer)
mongodb_thread = threading.Thread(target=mongodb_lookup)

# Start all threads
camera_thread.start()
extractor_thread.start()
analyzer_thread.start()
mongodb_thread.start()

# Join threads to ensure they complete
camera_thread.join()
extractor_thread.join()
analyzer_thread.join()
mongodb_thread.join()

cv2.destroyAllWindows()
