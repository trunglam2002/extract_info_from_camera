import cv2
import numpy as np
from fer import FER
from keras_facenet import FaceNet
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import logging
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import faiss
from typing import Sequence, Tuple
import tensorflow as tf

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info("GPU is available and will be used.")
else:
    logger.info("No GPU found. Using CPU.")

# Connect to MongoDB Atlas
uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]
collection = db["user_info"]

# Frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Cache for recognition results
recognition_cache = {}

# Load known face encodings
known_face_encodings = []
known_face_names = []
index = None  # Chỉ mục FAISS

# Khởi tạo MTCNN
detector = MTCNN()

# Tải mô hình cho anti-spoofing
model = load_model('model/mobilenetv2-epoch_15.hdf5')

# Biên dịch mô hình với bộ tối ưu và các chỉ số
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


def load_models():
    global emotion_detector, embedder
    # Initialize FER for emotion recognition
    emotion_detector = FER(mtcnn=False)
    # Initialize FaceNet
    embedder = FaceNet()


def load_face_database():
    global known_face_encodings, known_face_names
    if os.path.exists("face_database.pkl"):
        try:
            with open("face_database.pkl", "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    # FAISS yêu cầu dữ liệu dạng numpy array
                    known_face_encodings = np.array(list(data.values()))
                    known_face_names = list(data.keys())
                    # Xây dựng chỉ mục FAISS
                    build_faiss_index(known_face_encodings)
                else:
                    logger.error(
                        "Loaded data is not in the expected format (dict).")
        except Exception as e:
            logger.error(f"Error loading face database: {e}")
    else:
        logger.warning("Face database not found. Please create one.")


def build_faiss_index(face_encodings):
    global index
    dimension = face_encodings.shape[1]  # Số chiều của vector encoding
    index = faiss.IndexFlatL2(dimension)  # Sử dụng L2 distance
    index.add(face_encodings)  # Thêm tất cả các face encodings vào chỉ mục


load_models()
load_face_database()


def draw_bounding_box(frame, x, y, w, h, name, emotion, is_real):
    # Draw bounding box and display info
    color = (0, 255, 0) if is_real else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{name} | {emotion} | Real: {is_real}",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def preprocess_face(face_image, required_size=(160, 160)):
    # Check if the face_image is valid
    if face_image is None or face_image.size == 0:
        logger.error("Empty face image provided for preprocessing.")
        return None

    face_image = cv2.resize(face_image, required_size)
    return face_image


def recognize_face_faiss(face_encoding):
    if len(known_face_encodings) == 0:
        return "Unknown"

    # FAISS tìm kiếm ANN, trả về chỉ số và khoảng cách
    # Tìm kiếm với top 1 kết quả gần nhất
    D, I = index.search(np.array([face_encoding]), 1)

    # Ngưỡng để xác định nhận diện có thành công hay không
    SIMILARITY_THRESHOLD = 0.6
    if D[0][0] < SIMILARITY_THRESHOLD:  # FAISS trả về khoảng cách nhỏ hơn là tốt hơn
        return known_face_names[I[0][0]]  # Trả về tên tương ứng với chỉ số
    else:
        return "Unknown"


def analyze_face(frame, sequence_coordinate):
    # Detect emotions using pre-detected face rectangles
    emotions = emotion_detector.detect_emotions(
        frame, face_rectangles=sequence_coordinate)
    try:
        if emotions:  # Check if any emotions are detected
            dominant_emotion = max(
                emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            score = emotions[0]['emotions'][dominant_emotion]
            if score > 0.3:
                return dominant_emotion
            else:
                return "Neutral"
        return "Unknown"
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return "Unknown"


def anti_spoofing(face_image):
    # Tiền xử lý khung hình
    img = cv2.resize(face_image, (224, 224))
    img = img / 255.0  # Chia cho 255 để chuẩn hóa về [0, 1]
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch

    # Dự đoán
    predictions = model.predict(img)

    # Kiểm tra xác suất và gán nhãn dựa trên ngưỡng 0.5
    return predictions[0][0] < 0.5  # Nhãn True nếu xác suất lớn hơn 0.5 (real)


def process_face_detection(frame):
    # Detect faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)
    return results


def process_face_recognition(face_image):
    preprocessed_face = preprocess_face(face_image)
    face_encoding = embedder.embeddings([preprocessed_face])[0]
    # Sử dụng FAISS để nhận diện khuôn mặt
    name = recognize_face_faiss(face_encoding)
    return name


def process_emotion_detection(frame, sequence_coordinate):
    emotion = analyze_face(frame, sequence_coordinate)
    return emotion


def process_anti_spoofing(face_image):
    is_real = anti_spoofing(face_image)
    return is_real


def query_user_info(name):
    user_info = collection.find_one({"name": name})  # Truy vấn theo tên
    if user_info:
        return user_info  # Trả về thông tin người dùng
    else:
        return None  # Không tìm thấy người dùng


def preprocess_coordinate(x, y, w, h):
    face_rectangles: Sequence[Tuple[int, int, int, int]] = []
    # Thêm hình chữ nhật vào danh sách
    face_rectangles.append((x, y, w, h))  # Thêm tuple (x, y, w, h)
    return face_rectangles


def process_frame(frame, frame_count, skip_frames=5, previous_faces=None):
    if previous_faces is None:
        previous_faces = []

    if frame_count % skip_frames != 0:
        # Draw previous bounding boxes
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
                logger.info(f"User Info: {user_info}")
            x, y, w, h = face_boxes[i // 3]
            draw_bounding_box(frame, x, y, w, h, name, emotion, is_real)
            current_faces.append((x, y, w, h, name, emotion, is_real))

    return frame, current_faces


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    frame_count = 0
    previous_faces = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break

            processed_frame, previous_faces = process_frame(
                frame, frame_count, previous_faces=previous_faces)
            cv2.imshow('Camera Video', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
