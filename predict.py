import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model(
    "plant_disease_model.h5"
)

# Classes
classes = [
    "Potato Early Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

# Image path
image_path = input(
    "Enter image path: "
)

# Read image
img = cv2.imread(image_path)

# Resize
img = cv2.resize(
    img,
    (128,128)
)

# Normalize
img = img / 255.0

# Reshape
img = np.expand_dims(
    img,
    axis=0
)

# Prediction
prediction = model.predict(img)

predicted_class = np.argmax(
    prediction
)

confidence = np.max(
    prediction
) * 100

print("\nPrediction:")
print(classes[predicted_class])

print(
    f"Confidence: {confidence:.2f}%"
)