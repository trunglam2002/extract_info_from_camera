import cv2
from deepface import DeepFace
import pandas as pd

# Hàm vẽ hình chữ nhật và tên lên khuôn mặt


def draw_bounding_box(frame, x, y, w, h, name):
    # Vẽ hình chữ nhật quanh khuôn mặt
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị tên phía trên khuôn mặt
    cv2.putText(frame, name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


# Mở camera
cap = cv2.VideoCapture(0)

# Kiểm tra camera có mở được hay không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Ngưỡng khoảng cách để xác định người, ví dụ 0.6
distance_threshold = 0.6

# Sử dụng backend thích hợp (ví dụ sử dụng dlib hoặc retinaface)
backends = 'retinaface'  # Hoặc 'dlib', 'opencv', etc.

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Nếu không có khung hình nào, thoát vòng lặp

    # Chuyển đổi khung hình từ BGR sang RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện các khuôn mặt trong khung hình
    detected_faces = DeepFace.extract_faces(
        img_rgb, enforce_detection=False, detector_backend=backends)

    print(f"Số khuôn mặt phát hiện được: {len(detected_faces)}")

    if detected_faces is not None and len(detected_faces) > 0:
        # Duyệt qua từng khuôn mặt được phát hiện
        for i, detected_face in enumerate(detected_faces):
            facial_area = detected_face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            # Lưu khuôn mặt vào file tạm thời
            # Hình ảnh khuôn mặt đã phát hiện
            face_image = detected_face['face']

            # Đường dẫn đến cơ sở dữ liệu
            db_path = "test/my_db"

            # Tìm khuôn mặt trong cơ sở dữ liệu
            results = DeepFace.find(img_path=face_image, db_path=db_path,
                                    enforce_detection=False, detector_backend=backends)  # So sánh từng khuôn mặt

            # Kiểm tra xem results có chứa dữ liệu
            if results is not None and len(results) > 0:
                # Chuyển đổi danh sách kết quả thành DataFrame
                combined_results = pd.concat(results, ignore_index=True)

                if combined_results.empty:
                    print(
                        f"Không tìm thấy kết quả nào trong cơ sở dữ liệu cho khuôn mặt {i+1}.")
                    draw_bounding_box(frame, x, y, w, h, '')
                    continue

                # Lấy kết quả có distance thấp nhất cho khuôn mặt này
                # Tìm chỉ số của dòng có distance nhỏ nhất
                best_match_index = combined_results['distance'].idxmin()
                best_result = combined_results.iloc[best_match_index]

                print(best_result['distance'])

                # Truy cập cột 'identity'
                identity_path = best_result['identity']
                distance = best_result['distance']  # Truy cập cột 'distance'

                if distance < distance_threshold:
                    # Lấy tên từ đường dẫn thư mục
                    name = identity_path.split("/")[-2]
                    print(
                        f"Khuôn mặt {i+1}: Hiển thị tên {name} với khoảng cách {distance:.2f}")

                    # Vẽ hình chữ nhật và tên quanh khuôn mặt trong ảnh
                    draw_bounding_box(frame, x, y, w, h, name)
                else:
                    print(
                        f"Khuôn mặt {i+1}: Kết quả tìm thấy nhưng khoảng cách {distance:.2f} quá lớn, không hiển thị tên.")
                    draw_bounding_box(frame, x, y, w, h, '')
            else:
                print(
                    f"Khuôn mặt {i+1}: Không tìm thấy kết quả nào trong cơ sở dữ liệu.")
    else:
        print("Không phát hiện khuôn mặt trong khung hình.")

    # Hiển thị khung hình
    cv2.imshow('Video Output', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng camera/video
cap.release()
cv2.destroyAllWindows()

DeepFace.stream
