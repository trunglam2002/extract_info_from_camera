import cv2
import logging
import tensorflow as tf


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)


def setup_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def draw_bounding_box(frame, x, y, w, h, name, emotion, is_real):
    color = (0, 255, 0) if is_real else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{name} | {emotion} | Real: {is_real}",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def preprocess_coordinate(x, y, w, h):
    face_rectangles = []
    face_rectangles.append((x, y, w, h))
    return face_rectangles


def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info("GPU is available and will be used.")
    else:
        logging.info("No GPU found. Using CPU.")
