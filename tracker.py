import cv2
import numpy as np
from mtcnn import MTCNN
from fer import FER
from keras_facenet import FaceNet
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
from queue import Queue
from threading import Thread

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Tạo queue để lưu trữ thông tin debug
debug_queue = Queue()

# Connect to MongoDB Atlas
uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]
collection = db["user_info"]

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize FER for emotion recognition
emotion_detector = FER(mtcnn=True)

# Initialize FaceNet
embedder = FaceNet()

# Frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Cache for recognition results
recognition_cache = {}

# Load known face encodings
known_face_encodings = []
known_face_names = []


def process_debug_info():
    while True:
        info = debug_queue.get()
        if info is None:
            break
        logger.info(info)


# Khởi động thread xử lý debug
debug_thread = Thread(target=process_debug_info)
debug_thread.start()


def load_face_database():
    global known_face_encodings, known_face_names
    if os.path.exists("face_database.pkl"):
        with open("face_database.pkl", "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    else:
        debug_queue.put("Face database not found. Please create one.")


load_face_database()


def draw_bounding_box(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def analyze_face(face_image):
    try:
        emotions = emotion_detector.detect_emotions(face_image)

        if emotions:
            emotions_dict = emotions[0]['emotions']
            debug_queue.put(f"Detected emotions: {emotions_dict}")

            dominant_emotion = max(emotions_dict, key=emotions_dict.get)
            confidence = emotions_dict[dominant_emotion]

            debug_queue.put(
                f"Emotion: {emotions[0]['emotions']}, {confidence}")

            if confidence > 0.3:
                return dominant_emotion
            else:
                return "Neutral"
        return "Unknown"
    except Exception as e:
        debug_queue.put(f"Error in emotion analysis: {e}")
        return "Unknown"


def preprocess_face(face_image, required_size=(160, 160)):
    face_image = cv2.resize(face_image, required_size)
    face_image = face_image.astype('float32')
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    return face_image


def recognize_face(face_encoding):
    if len(known_face_encodings) == 0:
        return "Unknown"

    similarities = cosine_similarity([face_encoding], known_face_encodings)
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[0][best_match_index]

    # Đặt ngưỡng confidence (có thể điều chỉnh giá trị này)
    SIMILARITY_THRESHOLD = 0.6

    if best_match_score > SIMILARITY_THRESHOLD:
        return known_face_names[best_match_index]
    else:
        return "Unknown"


def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    for result in results:
        x, y, w, h = result['box']
        face_image = rgb_frame[y:y+h, x:x+w]

        preprocessed_face = preprocess_face(face_image)
        face_encoding = embedder.embeddings([preprocessed_face])[0]

        face_encoding_bytes = face_encoding.tobytes()

        if face_encoding_bytes in recognition_cache:
            name, emotion = recognition_cache[face_encoding_bytes]
        else:
            name = recognize_face(face_encoding)
            emotion = analyze_face(face_image)
            recognition_cache[face_encoding_bytes] = (name, emotion)

        debug_queue.put(f"Final result - Name: {name}, Emotion: {emotion}")

        if name != "Unknown":
            user_info = collection.find_one({"name": name})
            if user_info:
                info_text = f"User: {user_info['name']}, Emotion: {emotion}"
                debug_queue.put(
                    f"{info_text}, Phone: {user_info['phone']}, Account Number: {user_info['account_number']}")
            else:
                info_text = f"{name}, Emotion: {emotion}"
        else:
            info_text = f"Unknown, Emotion: {emotion}"

        draw_bounding_box(frame, x, y, w, h, info_text)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                future = executor.submit(process_frame, frame)
                processed_frame = future.result()

                cv2.imshow('Camera Video', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Kết thúc thread xử lý debug
        debug_queue.put(None)
        debug_thread.join()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
