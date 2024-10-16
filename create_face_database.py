import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import pickle
from collections import defaultdict

# Initialize FaceNet
embedder = FaceNet()

# Initialize MTCNN
detector = MTCNN()

face_data = defaultdict(list)

# Thư mục chứa ảnh khuôn mặt đã biết
face_directory = "known_faces"


def preprocess_face(face_image, required_size=(160, 160)):
    face_image = cv2.resize(face_image, required_size)
    face_image = face_image.astype('float32')
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    return face_image


for filename in os.listdir(face_directory):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(face_directory, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = detector.detect_faces(image_rgb)
        if results:
            x, y, w, h = results[0]['box']
            # Mở rộng vùng khuôn mặt để bao gồm nhiều chi tiết hơn
            x, y = max(0, x-20), max(0, y-20)
            w, h = min(image_rgb.shape[1]-x, w +
                       40), min(image_rgb.shape[0]-y, h+40)
            face_image = image_rgb[y:y+h, x:x+w]
            preprocessed_face = preprocess_face(face_image)
            face_encoding = embedder.embeddings([preprocessed_face])[0]

            # Trích xuất tên từ tên file, bỏ qua số ở cuối
            name = '_'.join(os.path.splitext(filename)[0].split('_')[:-1])
            face_data[name].append(face_encoding)
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {filename}")

# Tính trung bình các encoding cho mỗi người
known_face_encodings = []
known_face_names = []

for name, encodings in face_data.items():
    avg_encoding = np.mean(encodings, axis=0)
    known_face_encodings.append(avg_encoding)
    known_face_names.append(name)

# Lưu cơ sở dữ liệu vào file
with open("face_database.pkl", "wb") as f:
    pickle.dump({"encodings": known_face_encodings,
                "names": known_face_names}, f)

print("Face database created and saved.")
