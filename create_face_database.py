from collections import defaultdict
import pickle
import os
from keras_facenet import FaceNet
from mtcnn import MTCNN
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Initialize FaceNet
embedder = FaceNet()

# Initialize MTCNN
detector = MTCNN()

face_data = defaultdict(list)

# Thư mục chứa ảnh khuôn mặt đã biết
face_directory = "known_faces"


def preprocess_face(face_image, required_size=(160, 160)):
    face_image = cv2.resize(face_image, required_size)
    return face_image


def extract_name(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    if parts[-1].isdigit():
        parts = parts[:-1]
    return '_'.join(parts)


try:
    for filename in os.listdir(face_directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(face_directory, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to read image {filename}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print(f"Debug - Processing image: {filename}")
            print(f"Debug - Image shape: {image_rgb.shape}")

            results = detector.detect_faces(image_rgb)
            if results:
                x, y, w, h = results[0]['box']
                print(f"Debug - Face detected at: x={x}, y={y}, w={w}, h={h}")

                face_image = image_rgb[y:y+h, x:x+w]

                # Preprocess the face (resize and normalize)
                preprocessed_face = preprocess_face(face_image)
                face_encoding = embedder.embeddings([preprocessed_face])[0]

                print(f"Debug - Face encoding shape: {face_encoding.shape}")
                print(f"Debug - Face encoding[:5]: {face_encoding[:5]}")

                name = extract_name(filename)
                # Lưu encoding mà không cần filename
                face_data[name].append(face_encoding)
            else:
                print(f"Không tìm thấy khuôn mặt trong ảnh: {filename}")

    print("\nDebug - Final database information:")
    for name, encodings in face_data.items():
        print(f"Debug - Name: {name}")
        print(f"Debug - Number of encodings: {len(encodings)}")

        # Sử dụng K-means để tìm centroid cho các encoding
        kmeans = KMeans(n_clusters=1)  # Chỉ cần 1 centroid cho mỗi người
        kmeans.fit(encodings)
        mean_encoding = kmeans.cluster_centers_[0]

        print(f"Debug - Mean encoding shape: {mean_encoding.shape}")
        print(f"Debug - Mean encoding[:5]: {mean_encoding[:5]}")

        # Lưu encoding trung bình vào danh sách
        face_data[name] = mean_encoding

    # Lưu cơ sở dữ liệu vào file
    with open("face_database.pkl", "wb") as f:
        # Lưu dữ liệu dưới dạng dictionary
        pickle.dump(face_data, f)

    print("\nFace database created and saved.")

except Exception as e:
    print(f"An error occurred: {str(e)}")

# Kiểm tra database sau khi lưu
try:
    with open("face_database.pkl", "rb") as f:
        loaded_data = pickle.load(f)
        known_face_encodings = list(loaded_data.values())
        known_face_names = list(loaded_data.keys())

    print("\nDebug - Loaded database information:")
    print(f"Loaded encodings count: {len(known_face_encodings)}")
    print(f"Loaded names count: {len(known_face_names)}")

    # Hiển thị 5 encoding đầu tiên và tên tương ứng
    for i in range(min(5, len(known_face_encodings))):
        print(f"Debug - Name: {known_face_names[i]}")
        print(f"Debug - Encoding[:5]: {known_face_encodings[i][:5]}")

except Exception as e:
    print(f"Error loading database: {str(e)}")
