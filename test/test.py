
import numpy as np
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import pickle
import os
import matplotlib.pyplot as plt

# Initialize FaceNet
embedder = FaceNet()

# Load known face encodings
known_face_encodings = []
known_face_names = []

# Cấu hình
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SIMILARITY_THRESHOLD = 0.4


def load_face_database():
    global known_face_encodings, known_face_names
    if os.path.exists("face_database.pkl"):
        with open("face_database.pkl", "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
        print(f"Loaded {len(known_face_names)} faces from database.")
        print(f"Names in database: {known_face_names}")
        if known_face_encodings:  # Kiểm tra nếu danh sách không trống
            print("Debug - known_face_encodings shape:",
                  np.array(known_face_encodings).shape)
            print(
                "Debug - known_face_encodings[0][:5]:", known_face_encodings[0][:5])
        else:
            print("No face encodings found in the database.")
    else:
        print("Face database not found. Please create one.")


# def preprocess_face(face_image, required_size=(160, 160)):
#     face_image = cv2.resize(face_image, required_size)
#     face_image = face_image.astype('float32')
#     mean, std = face_image.mean(), face_image.std()
#     face_image = (face_image - mean) / std
#     return face_image


def recognize_face(face_encoding, similarity_threshold=0.6):
    best_match = ("Unknown", 0)
    if not known_face_encodings:  # Check if known_face_encodings is empty
        print("No known face encodings available for recognition.")
        return best_match  # Return default value
    similarities = cosine_similarity([face_encoding], known_face_encodings)[0]
    print(similarities)
    for name, similarity in zip(known_face_names, similarities):
        if similarity > best_match[1] and similarity > similarity_threshold:
            best_match = (name, similarity)
    return best_match


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(
        rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected in the image.")
        return

    for (x, y, w, h) in faces:  # Lặp qua tất cả các khuôn mặt được phát hiện
        face_image = rgb_image[y:y+h, x:x+w]

        # face_image = preprocess_face(face_image)
        face_encoding = embedder.embeddings([face_image])[0]

        print("Debug - face_encoding shape:", face_encoding.shape)
        print("Debug - face_encoding[:5]:", face_encoding[:5])

        name, confidence = recognize_face(face_encoding)
        print(f"Recognized: {name}, Confidence: {confidence:.4f}")

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{name} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    load_face_database()

    test_images = [
        "test/my_db/Obama/170305143551-trump-obama-split.jpg",
        "test/my_db/Obama/obama-jpg-9958-1452654475-1-1106.jpg",
        "test/my_db/Obama/Donald_Trump_official_portrait.jpg"
    ]

    for image_path in test_images:
        print(f"\nProcessing image: {image_path}")
        process_image(image_path)
