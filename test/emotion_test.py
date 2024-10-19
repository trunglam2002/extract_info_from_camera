import cv2
from fer import FER
from mtcnn import MTCNN
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize FER for emotion recognition
emotion_detector = FER(mtcnn=True)


def analyze_face(face_image):
    try:
        emotions = emotion_detector.detect_emotions(face_image)

        if emotions:
            emotions_dict = emotions[0]['emotions']
            logger.info(f"Detected emotions: {emotions_dict}")

            dominant_emotion = max(emotions_dict, key=emotions_dict.get)
            confidence = emotions_dict[dominant_emotion]

            logger.info(f"Emotion: {emotions[0]['emotions']}, {confidence}")

            if confidence > 0.3:
                return dominant_emotion
            else:
                return "Neutral"
        return "Unknown"
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return "Unknown"


def process_images(image_directory):
    for filename in os.listdir(image_directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Error: Unable to read image {filename}")
                continue

            # Convert the image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            results = detector.detect_faces(rgb_image)

            for result in results:
                x, y, w, h = result['box']
                face_image = rgb_image[y:y+h, x:x+w]

                # Analyze the face for emotions
                emotion = analyze_face(face_image)
                logger.info(f"Detected Emotion for {filename}: {emotion}")

                # Optionally, draw bounding box on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Display the image with detected faces and emotions
            cv2.imshow('Emotion Recognition', image)
            cv2.waitKey(0)  # Wait for a key press to move to the next image

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify the directory containing images
    # Change this to your image directory
    image_directory = "test/my_db/Obama"
    process_images(image_directory)
