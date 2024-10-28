import streamlit as st
import pickle
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Preprocess the image for model prediction
def preprocess_image(image):
    # Convert the image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image as per your model input shape (300x300 in this case)
    image = image.resize((300, 300))
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Normalize the image (if required by your model)
    img_array = img_array / 255.0
    
    # Reshape to add batch dimension (1, 300, 300, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict using the model
def predict_image(image):
    processed_image = preprocess_image(image)
    
    # Get prediction (assuming the model returns a single prediction)
    prediction = model.predict(processed_image)
    
    # Use argmax to get the predicted class label
    predicted_label = np.argmax(prediction, axis=1)  # Assuming binary classification

    # Check the predicted label and return the appropriate result
    if predicted_label[0] == 0:
        return "Good"
    else:
        return "Tampered"

# Streamlit app layout
st.title("Image Prediction App")
st.write("Upload a photo or capture using webcam to check if it's good or tampered.")

# Upload or capture photo feature
option = st.radio("Choose input method", ("Upload Photo", "Capture from Webcam"))

if option == "Upload Photo":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        result = predict_image(image)
        st.write(f"Prediction: {result}")

elif option == "Capture from Webcam":
    cap = st.camera_input("Capture Image")
    if cap is not None:
        # Read the image from the captured input
        image = Image.open(BytesIO(cap.getvalue()))
        st.image(image, caption="Captured Image", use_column_width=True)
        result = predict_image(image)
        st.write(f"Prediction: {result}")
