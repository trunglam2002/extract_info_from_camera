import cv2
import numpy as np
from fer import FER
from keras_facenet import FaceNet
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
from queue import Queue
from threading import Thread
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import faiss

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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

    # detector = MTCNN()


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


def analyze_face(emotion):
    try:
        if isinstance(emotion, dict):  # Check if emotion is a dictionary
            emo, score = emotion.items(), max(emotion.values())
            dominant_emotion = max(emotion, key=emotion.get)
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
    return predictions[0][0] > 0.5  # Nhãn True nếu xác suất lớn hơn 0.5 (real)


def process_face_detection(frame):
    # Detect faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = emotion_detector.detect_emotions(rgb_frame)
    return results


def process_face_recognition(face_image):
    preprocessed_face = preprocess_face(face_image)
    face_encoding = embedder.embeddings([preprocessed_face])[0]
    # Sử dụng FAISS để nhận diện khuôn mặt
    name = recognize_face_faiss(face_encoding)
    return name


def process_emotion_detection(emotion):
    emotion = analyze_face(emotion)
    return emotion


def process_anti_spoofing(face_image):
    is_real = anti_spoofing(face_image)
    return is_real


def process_frame(frame):
    results = process_face_detection(frame)
    with ThreadPoolExecutor() as executor:
        futures = []
        face_boxes = []  # Danh sách để lưu trữ thông tin khuôn mặt

        for result in results:
            x, y, w, h = result['box']
            face_image = frame[y:y + h, x:x + w]

            # Tạo các tác vụ cho các luồng riêng biệt
            futures.append(executor.submit(
                process_face_recognition, face_image))
            # Đảm bảo truyền đúng dữ liệu
            futures.append(executor.submit(
                process_emotion_detection, result['emotions']))
            futures.append(executor.submit(process_anti_spoofing, face_image))

            # Lưu trữ thông tin khuôn mặt
            face_boxes.append((x, y, w, h))

        # Gọi hàm `draw_bounding_box` sau khi tất cả các kết quả từ các luồng đều đã hoàn thành
        for i in range(0, len(futures), 3):
            name = futures[i].result()
            emotion = futures[i + 1].result()
            is_real = futures[i + 2].result()

            # Vẽ chữ cho từng khuôn mặt tại vị trí tương ứng
            # Lấy thông tin khuôn mặt tương ứng
            x, y, w, h = face_boxes[i // 3]
            draw_bounding_box(frame, x, y, w, h, name, emotion, is_real)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break

            # Xử lý frame với các luồng riêng biệt
            processed_frame = process_frame(frame)

            # Hiển thị frame đã xử lý
            cv2.imshow('Camera Video', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
