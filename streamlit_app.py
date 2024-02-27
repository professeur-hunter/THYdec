import streamlit as st
import tensorflow
from PIL import Image, ImageOps
#from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import keras
st.title("THYDet")
st.header("Ultrasound Thyroid Detector")
st.text("Upload an Image.....")

from PIL import Image
from keras.utils import load_img,img_to_array
import numpy as np
import keras
from keras.models import load_model
from PIL import Image, ImageOps
def teachable_machine_classification(img, weights_file):
    
    model = keras.models.load_model(weights_file)

    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
  
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

  
    image_array = np.asarray(image)
  
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

  
    data[0] = normalized_image_array

    prediction = model.predict(data)
    return np.argmax(prediction)
     
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    st.write("Guess")
    label = teachable_machine_classification(image, 'resnet.h5')
    if label == 0:
       st.write("TIRAD1")
    elif label == 1:
       st.write("TIRAD2")
    elif label == 2:
       st.write("TIRAD3")
    elif label == 3:
       st.write("TIRAD4A")
    elif label == 4:
       st.write("TIRAD4B")
    elif label == 5:
       st.write("TIRAD4C")
    else:
       st.write("TIRAD5")
   
        
        
