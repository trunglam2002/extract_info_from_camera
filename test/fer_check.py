import cv2
import os
from fer import FER

# Khởi tạo mô hình FER
emotion_detector = FER()

def detect_and_display_emotions(folder_path):
    # Duyệt qua tất cả các tệp trong thư mục
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Kiểm tra định dạng tệp ảnh
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")  # Thêm thông báo debug
            
            # Đọc ảnh
            image = cv2.imread(image_path)
            
            # Phát hiện khuôn mặt
            detections = emotion_detector.detect_emotions(image)
            
            # Nếu có khuôn mặt được phát hiện
            if detections:
                print(detections[0])
                # Lấy khuôn mặt đầu tiên
                face = detections[0]['box']
                
                # Crop khuôn mặt
                x, y, w, h = face
                face_cropped = image[y:y+h, x:x+w]
                
                # Resize khuôn mặt về kích thước 64x64
                face_resized = cv2.resize(face_cropped, (64, 64))
                
                # Dự đoán cảm xúc
                top_emotion, score = emotion_detector.top_emotion(face_resized)
                label = f"{top_emotion}: {score:.2f}"
                
                # Hiển thị nhãn cảm xúc trên ảnh
                cv2.putText(face_resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Hiển thị ảnh
                cv2.imshow(f"Emotion Detection - {filename}", face_resized)
                cv2.waitKey(0)  # Chờ nhấn phím để tiếp tục
                cv2.destroyAllWindows()

# Sử dụng hàm
folder_path = 'test/my_db/emotion'  # Thay đổi đường dẫn đến thư mục chứa ảnh
detect_and_display_emotions(folder_path)