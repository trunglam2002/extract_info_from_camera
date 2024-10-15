from deepface import DeepFace
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

alignment_modes = [True, False]

demographies = DeepFace.analyze(
    img_path="test/img/Trump_1.png",
    actions='emotion',
    detector_backend=backends[3],
    align=alignment_modes[0],
)

print(demographies)
