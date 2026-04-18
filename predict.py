import os
import gdown
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

# MODEL_PATH = hf_hub_download(
#     repo_id="your-username/plant-model",
#     filename="plant_model.h5"
# )
MODEL_PATH = "plant_model.h5"

# model = tf.keras.models.load_model(MODEL_PATH, compile=False)

file_id = "1MC6_8cqW_YSii6BD6YbyYBV3QLkkKloX"
url = f"https://drive.google.com/uc?id={file_id}"
if not os.path.exists("plant_model.h5"):
    print("Downloading model...")
    gdown.download(url, "plant_model.h5", quiet=False)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
class_names = [
    "Apple___Cedar_apple_rust",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Pepper___healthy"
]

def predict_image(image):
    try:
        image = image.resize((224, 224))
        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        index = np.argmax(pred)

        return class_names[index], float(np.max(pred))

    except Exception as e:
        return "Error", 0.0
