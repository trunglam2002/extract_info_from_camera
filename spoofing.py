import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tải mô hình
model = load_model('model/mobilenetv2-epoch_15.hdf5')

# Biên dịch mô hình với bộ tối ưu và các chỉ số
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Khởi động camera
cap = cv2.VideoCapture(0)  # 0 là chỉ số camera mặc định

# Tải cascade phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển đổi sang ảnh xám
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt
        face = frame[y:y+h, x:x+w]
        # Tiền xử lý khung hình
        img = cv2.resize(face, (224, 224))
        img = img / 255.0  # Chia cho 255 để chuẩn hóa về [0, 1]
        img = np.expand_dims(img, axis=0)  # Thêm chiều batch

        # Dự đoán
        predictions = model.predict(img)

        # Kiểm tra xác suất và gán nhãn dựa trên ngưỡng 0.5
        if predictions[0][0] > 0.5:
            class_label = 1  # Nhãn 1 nếu xác suất lớn hơn 0.5
        else:
            class_label = 0  # Nhãn 0 nếu xác suất nhỏ hơn hoặc bằng 0.5

        # Hiển thị kết quả
        cv2.putText(frame, f'Class: {class_label}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
