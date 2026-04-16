import streamlit as st
from PIL import Image
from predict import predict_image   # ✅ IMPORTANT (not train_model)

# Disease information
disease_info = {
    "Apple___Cedar_apple_rust": {
        "cause": "Fungal infection due to humid conditions",
        "effect": "Leaf spots, early leaf drop, reduced yield",
        "treatment": "Apply fungicide, remove infected leaves"
    },
    "Tomato___Late_blight": {
        "cause": "Fungal infection due to high humidity",
        "effect": "Leaf damage, crop loss",
        "treatment": "Apply fungicide, remove infected leaves"
    },
    "Potato___Early_blight": {
        "cause": "Fungus Alternaria",
        "effect": "Brown spots, reduced yield",
        "treatment": "Crop rotation, fungicide spray"
    },
    "Pepper___healthy": {
        "cause": "No disease",
        "effect": "Healthy plant",
        "treatment": "Maintain proper care"
    }
}

# UI starts here
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image and click Predict 👇")

# File upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# If no file uploaded
if uploaded_file is None:
    st.warning("Please upload an image to continue")

# If file uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        try:
            st.write("🔍 Predicting...")   # Debug line

            disease, confidence = predict_image(image)

            st.success(f"🌱 Disease: {disease}")
            st.info(f"Confidence: {confidence*100:.2f}%")

            # Show disease info
            if disease in disease_info:
                info = disease_info[disease]

                st.subheader("🌿 Disease Details")
                st.write("Cause:", info["cause"])
                st.write("Effect:", info["effect"])
                st.write("Treatment:", info["treatment"])
            else:
                st.warning("No additional info available")

        except Exception as e:
            st.error(f"Error: {e}")