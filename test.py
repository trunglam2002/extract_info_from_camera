# face_emotion_detection.py

import cv2
from fer import FER

# Khởi tạo FER với MTCNN
emotion_detector = FER(mtcnn=True)  # Sử dụng MTCNN từ FER


def detect_faces_and_emotions(frame):
    # Phát hiện khuôn mặt và cảm xúc
    emotions = emotion_detector.detect_emotions(frame)

    return emotions


def main():
    cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break

        # Phát hiện khuôn mặt và cảm xúc
        emotions = detect_faces_and_emotions(frame)

        # Vẽ hộp bao quanh khuôn mặt và hiển thị cảm xúc
        for result in emotions:
            x, y, w, h = result['box']
            emotion, score = result['emotions'].items(), max(
                result['emotions'].values())
            dominant_emotion = max(
                result['emotions'], key=result['emotions'].get)

            color = (0, 255, 0)  # Màu xanh cho khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{dominant_emotion}: {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Hiển thị khung hình
        cv2.imshow('Face and Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
