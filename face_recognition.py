import numpy as np
from keras_facenet import FaceNet
import faiss
import os
import pickle
import logging
import cv2

embedder = FaceNet()
known_face_encodings = []
known_face_names = []
index = None


def load_face_database():
    global known_face_encodings, known_face_names
    if os.path.exists("face_database.pkl"):
        try:
            with open("face_database.pkl", "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    known_face_encodings = np.array(list(data.values()))
                    known_face_names = list(data.keys())
                    build_faiss_index(known_face_encodings)
                else:
                    logging.error(
                        "Loaded data is not in the expected format (dict).")
        except Exception as e:
            logging.error(f"Error loading face database: {e}")
    else:
        logging.warning("Face database not found. Please create one.")


def build_faiss_index(face_encodings):
    global index
    dimension = face_encodings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(face_encodings)


def preprocess_face(face_image, required_size=(160, 160)):
    if face_image is None or face_image.size == 0:
        logging.error("Empty face image provided for preprocessing.")
        return None

    face_image = cv2.resize(face_image, required_size)
    return face_image


def recognize_face_faiss(face_encoding):
    if len(known_face_encodings) == 0:
        return "Unknown"

    D, I = index.search(np.array([face_encoding]), 1)
    SIMILARITY_THRESHOLD = 0.6
    if D[0][0] < SIMILARITY_THRESHOLD:
        return known_face_names[I[0][0]]
    else:
        return "Unknown"


def process_face_recognition(face_image):
    preprocessed_face = preprocess_face(face_image)
    face_encoding = embedder.embeddings([preprocessed_face])[0]
    name = recognize_face_faiss(face_encoding)
    return name


load_face_database()
