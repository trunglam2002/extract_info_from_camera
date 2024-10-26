import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/mobilenetv2-epoch_15.hdf5')
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


def anti_spoofing(face_image):
    img = cv2.resize(face_image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    return predictions[0][0] < 0.5


def process_anti_spoofing(face_image):
    is_real = anti_spoofing(face_image)
    return is_real
