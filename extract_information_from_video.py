import cv2
from deepface import DeepFace
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import threading
import queue

# Define the backend and MongoDB connection
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
            'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']
backend = backends[7]
alignment_modes = [True, False]

uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]
collection = db["user_info"]

# Function to draw bounding box and display information on the face


def draw_bounding_box(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


# Initialize queues for inter-thread communication
frame_queue = queue.Queue()
result_queue = queue.Queue()

# Thread for reading video frames


def frame_reader(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)  # Indicate end of video

# Thread for analyzing faces


def face_analyzer():
    while True:
        frame = frame_queue.get()
        if frame is None:  # Check for end of video
            result_queue.put((None, None))
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = DeepFace.extract_faces(
            img_rgb, enforce_detection=False, detector_backend=backend)
        if detected_faces:
            results = []
            for detected_face in detected_faces:
                face_image = detected_face['face']
                emotions = DeepFace.analyze(face_image, actions=[
                                            'emotion'], enforce_detection=False, detector_backend=backend, align=alignment_modes[0])
                results.append((detected_face, emotions))
            result_queue.put((frame, results))
        else:
            result_queue.put((frame, None))

# Thread for processing results and drawing bounding boxes


def result_processor(output_path):
    # Get video properties for output
    frame_count = 0
    distance_threshold = 0.6
    db_path = "test/my_db"

    # Create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Adjust frame size as necessary
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

    while True:
        result = result_queue.get()
        if result[0] is None:  # Check for end of analysis
            break

        frame, detected_faces_results = result

        # Process every 5th frame
        if frame_count % 5 == 0:
            if detected_faces_results is not None:
                for detected_face, emotions in detected_faces_results:
                    facial_area = detected_face['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    emotion_text = emotions[0]['dominant_emotion'] if emotions else 'Unknown'

                    # Compare face with database
                    results = DeepFace.find(
                        img_path=detected_face['face'], db_path=db_path, enforce_detection=False, detector_backend=backend, silent=True)

                    if results and len(results) > 0:
                        combined_results = pd.concat(
                            results, ignore_index=True)

                        if not combined_results.empty:
                            best_match_index = combined_results['distance'].idxmin(
                            )
                            best_result = combined_results.iloc[best_match_index]
                            identity_path = best_result['identity'].replace(
                                "\\", "/")
                            distance = best_result['distance']

                            if distance < distance_threshold:
                                name = identity_path.split("/")[-2]
                                user_info = collection.find_one({"name": name})
                                if user_info:
                                    user_text = f"{user_info['name']} - phone: {user_info['phone']}, account_number: {user_info['account_number']}, emotion: {emotion_text}"
                                    draw_bounding_box(
                                        frame, x, y, w, h, f"{name} - {emotion_text}")
                                else:
                                    draw_bounding_box(
                                        frame, x, y, w, h, f"{name} - {emotion_text}")
                            else:
                                draw_bounding_box(
                                    frame, x, y, w, h, f"Unknown - {emotion_text}")
                    else:
                        draw_bounding_box(frame, x, y, w, h,
                                          f"Unknown - {emotion_text}")
            else:
                print("No faces detected.")

        # Write the processed frame to output
        out.write(frame)
        frame_count += 1

    out.release()
    cv2.destroyAllWindows()


# Define video path and output path
video_path = "Obama_vid.mp4"
output_path = "output_with_info.mp4"

# Create and start threads
reader_thread = threading.Thread(target=frame_reader, args=(video_path,))
analyzer_thread = threading.Thread(target=face_analyzer)
processor_thread = threading.Thread(
    target=result_processor, args=(output_path,))

reader_thread.start()
analyzer_thread.start()
processor_thread.start()

# Wait for threads to finish
reader_thread.join()
analyzer_thread.join()
processor_thread.join()

print(f"Video đã được lưu thành công vào {output_path}")
