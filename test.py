from tensorflow.keras.models import load_model

# Load the FaceNet Keras model
facenet_model = load_model('model/facenet_keras.h5')
# Verify the model's architecture
model.summary()
