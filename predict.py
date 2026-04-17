import os
import gdown
import numpy as np
import tensorflow as tf

MODEL_PATH = "plant_model.h5"

file_id = "1MC6_8cqW_YSii6BD6YbyYBV3QLkkKloX"
url = f"https://drive.google.com/uc?id={file_id}"

# download safely
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(url, MODEL_PATH, quiet=False)

# CHECK if file exists BEFORE loading
if not os.path.exists(MODEL_PATH):
    raise Exception("Model download failed!")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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

    return class_names[index], float(np.max(pred))
