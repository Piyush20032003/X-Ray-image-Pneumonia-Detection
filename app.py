import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.preprocessing import image

# Loading the trained model
model = keras.models.load_model('model.h5')

## Streamlit app
st.title('X-Ray Pneumonia Detection')



def classify_image(uploaded_file):
    # Preprocess the uploaded file
    testing_image = image.load_img(uploaded_file, target_size=(64, 64))
    testing_image = image.img_to_array(testing_image)
    testing_image = np.expand_dims(testing_image, axis=0)

    # Model prediction
    result = model.predict(testing_image)

    # Output classification result
    if result[0][0] == 0:
        st.write("The image is classified as: Normal")
    else:
        st.write("The image is classified as: Pneumonia")

# Example usage in Streamlit
uploaded_file = st.file_uploader("Choose an X-ray Image", type=[".jpg",".png",".jpeg"],accept_multiple_files=False)
if uploaded_file is not None:
    classify_image(uploaded_file)