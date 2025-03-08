import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Define class labels
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
CLASS_LABELS_EMOJIS = ["😡", "🤢", "😱", "😊", "😐", "😔", "😲"]

# Load trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("emotion_detection_model.h5")
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ff4b4b;
        }
        .subheader {
            text-align: center;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>😡🤢😱😊😐😔😲 Emotion Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload an image or use the camera to detect emotions!</p>", unsafe_allow_html=True)

# Webcam capture
use_camera = st.toggle("Use Camera")
if use_camera:
    camera_image = st.camera_input("Capture an image")
    uploaded_file = camera_image
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    image = image.resize((48, 48))
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = CLASS_LABELS[predicted_class]
    predicted_emoji = CLASS_LABELS_EMOJIS[predicted_class]
    
    st.markdown("""
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: #4caf50;">
            Prediction Result:
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h2 style='text-align: center;'>{predicted_label} {predicted_emoji}</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>Prediction Probabilities:</h3>", unsafe_allow_html=True)
    
    for i, (label, emoji) in enumerate(zip(CLASS_LABELS, CLASS_LABELS_EMOJIS)):
        st.progress(float(prediction[0][i]))
        st.write(f"{label} {emoji}: {prediction[0][i]*100:.2f}%")
