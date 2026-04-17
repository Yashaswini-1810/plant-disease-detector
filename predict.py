import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(
    "plant_model.h5",
    compile=False
)
class_names = [
    "Apple___Cedar_apple_rust",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Pepper___healthy"
]

def predict_image(image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = np.argmax(pred)

    return class_names[index], np.max(pred)
