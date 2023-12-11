# import the Package
import os

import numpy as np 
from PIL import Image , ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf

LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Create a function to load my saved model
@st.cache_data()
def load_my_model():
    model = tf.keras.models.load_model("cifar10model.h5")
    return model

model = load_my_model()

# Create a title of web App
st.title("Cifar-10 Dataset Image Classification ")
st.header("Please Upload images related to following.")
st.write(LABELS)

# create a file uploader and take a image as an jpg or png
file = st.file_uploader("Upload the image" , type=["jpg" , "png"])

# Create a function to take and image and predict the class
def import_and_predict(image_data , model):
    size = (32 ,32)
    image = ImageOps.fit(image_data , size , Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if st.button("Predict"):
    image = Image.open(file)
    st.image(image , use_column_width=True)
    predictions = import_and_predict(image , model)

    LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    string = "I'm sure you have uploaded :-" + LABELS[np.argmax(predictions)]
    st.success(string)
    
    
