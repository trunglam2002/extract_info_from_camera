import cv2
from deepface import DeepFace
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'fastmtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]

backend = backends[7]
alignment_modes = [True, False]

# Kết nối MongoDB Atlas
uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]  # Cơ sở dữ liệu MongoDB
collection = db["user_info"]  # Collection MongoDB


# Hàm vẽ hình chữ nhật và hiển thị tên, thông tin, cảm xúc trên khuôn mặt
def draw_bounding_box(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


# Đọc video đầu vào
video_path = "Obama_vid.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông tin về chiều rộng, chiều cao và FPS của video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo đối tượng VideoWriter để ghi video đầu ra
output_path = "output_with_info.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Ngưỡng khoảng cách để xác định người, ví dụ 0.6
distance_threshold = 0.6

# Đường dẫn đến cơ sở dữ liệu khuôn mặt
db_path = "test/my_db"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected_faces = DeepFace.extract_faces(
        img_rgb, enforce_detection=False, detector_backend=backend)

    print(f"Số khuôn mặt phát hiện được: {len(detected_faces)}")

    if detected_faces is not None and len(detected_faces) > 0:
        for i, detected_face in enumerate(detected_faces):
            facial_area = detected_face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            face_image = detected_face['face']

            # Nhận diện cảm xúc
            emotions = DeepFace.analyze(
                face_image, actions=['emotion'], enforce_detection=False, detector_backend=backend, align=alignment_modes[0])
            emotion_text = emotions[0]['dominant_emotion'] if emotions else 'Unknown'

            # So sánh khuôn mặt với dữ liệu trong db
            results = DeepFace.find(img_path=face_image, db_path=db_path,
                                    enforce_detection=False, detector_backend=backend, silent=True)

            if results is not None and len(results) > 0:
                combined_results = pd.concat(results, ignore_index=True)

                if not combined_results.empty:
                    best_match_index = combined_results['distance'].idxmin()
                    best_result = combined_results.iloc[best_match_index]

                    identity_path = best_result['identity']
                    identity_path = identity_path.replace("\\", "/")
                    distance = best_result['distance']

                    if distance < distance_threshold:
                        # Lấy tên từ thư mục
                        name = identity_path.split("/")[-2]

                        # Truy xuất thông tin từ MongoDB dựa vào tên
                        user_info = collection.find_one({"name": name})

                        if user_info:
                            user_text = f"{user_info['name']} - phone: {user_info['phone']}, account_number: {user_info['account_number']}, emotion: {emotion_text}"
                            print(f"Thông tin người dùng {i+1 }: {user_text}")
                            draw_bounding_box(
                                frame, x, y, w, h, f"{str(i+1)}: {user_info['name']} - {emotion_text}")
                        else:
                            print(f"Không tìm thấy thông tin cho {name}")
                            draw_bounding_box(
                                frame, x, y, w, h, f"{str(i+1)}: {name} - {emotion_text}")
                    else:
                        draw_bounding_box(frame, x, y, w, h,
                                          f"{str(i+1)} - {emotion_text}")
            else:
                print(f"Khuôn mặt {i+1}: Không tìm thấy kết quả nào.")
                draw_bounding_box(frame, x, y, w, h,
                                  f"{str(i+1)} - {emotion_text}")

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video đã được lưu thành công vào {output_path}")
