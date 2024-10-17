import cv2
from fer import FER
from queue import Queue
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Tạo queue để lưu trữ thông tin debug
debug_queue = Queue()

# Initialize FER for emotion recognition
emotion_detector = FER(mtcnn=True)


def process_debug_info():
    while True:
        info = debug_queue.get()
        if info is None:
            break
        logger.info(info)


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


if __name__ == "__main__":
    # Test the emotion recognition function
    # Capture a single frame from the webcam for testing
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Assume the face is detected and cropped (you can replace this with actual face detection)
        # Here we just use the whole frame for testing
        emotion = analyze_face(rgb_frame)
        print(f"Detected Emotion: {emotion}")
    else:
        print("Failed to capture image.")

    cap.release()
