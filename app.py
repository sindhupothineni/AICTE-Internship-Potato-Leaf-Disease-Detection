import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id="1bpAwaa4pNO8ywV6u4gUuBLKwghjJmwTN"
url='https://drive.google.com/uc?id=1bpAwaa4pNO8ywV6u4gUuBLKwghjJmwTN'
model_path="trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url,model_path,quiet=False)

model_path="trained_plant_disease_model.keras"
def model_prediction(test_image):
    model=tf.keras.models.load_model(model_path)
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])#convert single image to batch
    predictions=model.predict(input_arr)
    return np.argmax(predictions)
from PIL import Image
img=Image.open('Disease.jpg')
st.image(img,width=500) 

st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode=st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])

#Main page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align:center;'>Plant Disease Detection System for Sustainable Agriculture",unsafe_allow_html=True)
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image=st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index=model_prediction(test_image)
        #Reading Labels
        class_name=['Potato_Early_blight','Potato_Late_blight','Potato_Healthy']
        st.success("Model is predicting. Its a {}".format(class_name[result_index]))


