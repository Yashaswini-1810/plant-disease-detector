import tensorflow as tf
import numpy as np
import gdown
url = "https://colab.research.google.com/drive/1a09kIv6OaqFJBW8y38uu9hpQoqxHEs6g?usp=sharing"
gdown.download(url, "plant_model.h5", quiet=False)
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
