from fer import FER
import logging

emotion_detector = FER(mtcnn=False)


def analyze_face(frame, sequence_coordinate):
    emotions = emotion_detector.detect_emotions(
        frame, face_rectangles=sequence_coordinate)
    try:
        if emotions:
            dominant_emotion = max(
                emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            score = emotions[0]['emotions'][dominant_emotion]
            if score > 0.3:
                return dominant_emotion
            else:
                return "Neutral"
        return "Unknown"
    except Exception as e:
        logging.error(f"Error in emotion analysis: {e}")
        return "Unknown"


def process_emotion_detection(frame, sequence_coordinate):
    emotion = analyze_face(frame, sequence_coordinate)
    return emotion
