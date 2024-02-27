import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import keras
st.title("THYDet")
st.header("Ultrasound Thyroid Detector")
st.text("Upload an Image.....")
     
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    st.write("Guess")
    label = teachable_machine_classification(image, 'resnet.h5')
    if label == 0:
       st.write("It predicted Anaconda")
    else:
       st.write("It predicted Python")
   
        
        
