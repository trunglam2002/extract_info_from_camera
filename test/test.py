import os
import cv2
import numpy as np
import tensorflow as tf

# Kiểm tra GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Đường dẫn tới thư mục chứa dữ liệu
data_dir = '/kaggle/input/anti-spoofing-live/samples'

# Tham số
frame_count = 30  # Số khung hình muốn lấy từ video
sample_rate = 5   # Lấy 1 khung hình sau mỗi 5 khung hình

# Hàm xử lý video


def load_and_preprocess_video(video_path, label):
    cap = cv2.VideoCapture(video_path.numpy().decode(
        'utf-8'))  # Decode từ tensor sang string
    frames = []

    count = 0
    while cap.isOpened() and len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if count % sample_rate == 0:
            frame = cv2.resize(frame, (160, 160))
            frame = frame.astype(np.float32) / 255.0  # Chuẩn hóa
            frames.append(frame)

        count += 1

    cap.release()

    # Chắc chắn có 30 khung hình
    while len(frames) < frame_count:
        frames.append(np.zeros((160, 160, 3)))  # Thêm khung hình trắng nếu cần

    # Chuyển đổi danh sách khung hình thành numpy array
    frames = np.array(frames)
    # Trả về tensor
    return tf.convert_to_tensor(frames, dtype=tf.float32), label

# Hàm tải và xử lý ảnh selfie


def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [160, 160])
    image /= 255.0  # Hoặc tf.image.per_image_standardization(image)
    return image, label

# Hàm lấy danh sách đường dẫn và nhãn


def create_dataset(data_dir):
    image_paths = []
    video_paths = []

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)

        if os.path.isdir(folder_path):
            selfie_path = os.path.join(folder_path, 'live_selfie.jpg')
            video_path = os.path.join(folder_path, 'live_video.mp4')
            video_path_mov = os.path.join(
                folder_path, 'live_video.MOV')  # Đường dẫn cho file MOV

            if os.path.exists(selfie_path) and (os.path.exists(video_path) or os.path.exists(video_path_mov)):
                image_paths.append(selfie_path)

                # Kiểm tra tệp video
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                elif os.path.exists(video_path_mov):
                    video_paths.append(video_path_mov)

    return image_paths, video_paths


# Tạo danh sách đường dẫn
image_paths, video_paths = create_dataset(data_dir)

# Tạo một Dataset từ các đường dẫn và nhãn cho ảnh selfie
image_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, [0] * len(image_paths)))  # Gán nhãn 0 cho tất cả ảnh
image_dataset = image_dataset.map(
    load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Tạo một Dataset từ các đường dẫn và nhãn cho video
video_dataset = tf.data.Dataset.from_tensor_slices(
    (video_paths, [1] * len(video_paths)))  # Gán nhãn 1 cho tất cả video
video_dataset = video_dataset.map(lambda v_path, lbl: tf.py_function(load_and_preprocess_video, [
                                  v_path, lbl], [tf.float32, tf.int32]), num_parallel_calls=tf.data.AUTOTUNE)

# Đảm bảo rằng video có kích thước nhất quán
video_dataset = video_dataset.map(lambda video_tensor, label: (tf.reshape(
    video_tensor, (30, 160, 160, 3)), label), num_parallel_calls=tf.data.AUTOTUNE)

# Batch riêng cho mỗi dataset
image_dataset = image_dataset.batch(32)
video_dataset = video_dataset.batch(32)

# Kết hợp cả hai dataset
dataset = tf.data.Dataset.zip(
    (image_dataset, video_dataset)).shuffle(buffer_size=1000)

# Tối ưu hóa tải trước
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Kiểm tra kích thước của dataset
for data in dataset.take(1):
    print(
        f'Image Data shape: {data[0][0].shape}, Video Data shape: {data[1][0].shape}, Labels: {data[0][1].shape}')
