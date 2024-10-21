# Face Recognition and Anti-Spoofing Application

## Description
This application uses face recognition and anti-spoofing technology to detect and recognize user faces from a camera feed. The application consists of two main parts:
1. **Create Face Database**: Stores known face encodings into a file for later recognition.
2. **Extract Information from Camera**: Recognizes faces, analyzes emotions, and checks the authenticity of faces (anti-spoofing) in real time.

## Technologies Used
- **Python**: The main programming language for the application.
- **OpenCV**: Library for processing videos and images.
- **Keras**: Library for building and training deep learning models.
- **FaceNet**: Deep learning model for face recognition.
- **MTCNN**: Face detection model.
- **FER**: Library for emotion recognition from faces.
- **MongoDB**: Database for storing user information.
- **FAISS**: Facebook AI library for efficient vector search.
- **Pickle**: Library for storing and loading data.

## How to Run the Application

### Setting Up the Environment

1. **Install Python**: Ensure you have Python installed (version 3.6 or newer) on your system.
2. **Install Required Libraries**:
   You can use `pip` to install the necessary libraries. Optionally, create a virtual environment and install as follows:
   ```bash
   pip install opencv-python keras keras-facenet mtcnn fer pymongo faiss-cpu

   ## Face Recognition Application

### Create Face Database

1. **Prepare Images**: 
   - Place known face images into a folder named `known_faces` in the same directory as the source code.

2. **Run the Database Creation Script**: 
   - Execute the `create_face_database.py` script to create the face database:
     ```bash
     python create_face_database.py
     ```

### Extract Information from Camera

1. **Run the Camera Extraction Script**: 
   - Once you have the face database, run the `extract_info_from_camera.py` script to start recognizing faces from the camera:
     ```bash
     python extract_info_from_camera.py
     ```

2. **Stop the Application**: 
   - Press `q` to stop the face recognition application.
  
# Mục lục

1. [System Requirements](#system-requirements)
2. [How to Use](#how-to-use)
3. [Function Explanations](#function-explanations)
4. [Creating Threads with ThreadPoolExecutor](#creating-threads-with-threadpoolexecutor)
5. [Operational Diagram](#operational-diagram)
