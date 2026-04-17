import streamlit as st
from PIL import Image
from predict import predict_image

st.title("🌿 Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        disease, confidence = predict_image(image)

        st.success(f"Disease: {disease}")
        st.info(f"Confidence: {confidence*100:.2f}%")
