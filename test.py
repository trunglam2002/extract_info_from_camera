import cv2
from fer import FER
import mediapipe as mp  # Import the FER library
import numpy as np
from typing import Sequence, Tuple

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Sử dụng camera laptop
cap = cv2.VideoCapture(0)  # Thay đổi từ RTSP sang camera laptop
# Khởi tạo bộ nhận diện cảm xúc với OpenCV
emotion_detector = FER(mtcnn=False)


def preprocess_face(face_region, target_size=(48, 48), padding=40):
    """Xử lý khuôn mặt để chuẩn bị cho việc nhận diện cảm xúc."""
    # Chuyển đổi sang ảnh xám
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Thêm padding
    row, col = gray_face.shape[:2]
    padded_face = cv2.copyMakeBorder(
        gray_face,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    # Resize khuôn mặt
    resized_face = cv2.resize(padded_face, target_size)
    return resized_face


with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break

        # Chuyển đổi màu sắc từ BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # Danh sách chứa các hình chữ nhật khuôn mặt
        face_rectangles: Sequence[Tuple[int, int, int, int]] = []

        # Vẽ hộp bao quanh khuôn mặt và nhận diện cảm xúc
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin *
                                                       ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Thêm hình chữ nhật vào danh sách
                face_rectangles.append((x, y, w, h))  # Thêm tuple (x, y, w, h)

        # Nhận diện cảm xúc từ các khuôn mặt đã phát hiện
        if face_rectangles:
            # Truyền frame và face_rectangles vào hàm
            emotions = emotion_detector.detect_emotions(frame, face_rectangles)

            # Lấy cảm xúc với điểm số cao nhất cho từng khuôn mặt
            for i, (x, y, w, h) in enumerate(face_rectangles):
                if emotions and i < len(emotions):
                    top_emotion = max(
                        emotions[i]['emotions'], key=emotions[i]['emotions'].get)
                    score = emotions[i]['emotions'][top_emotion]
                    cv2.putText(frame, f'{top_emotion} {score:.2f}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, 'Unknown', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Hiển thị video
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
