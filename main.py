import cv2
import logging
from utils import setup_camera, setup_logging
from face_detection import process_frame


def main():
    setup_logging()
    cap = setup_camera()
    frame_count = 0
    previous_faces = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera.")
                break

            processed_frame, previous_faces = process_frame(
                frame, frame_count, previous_faces=previous_faces)
            cv2.imshow('Camera Video', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
