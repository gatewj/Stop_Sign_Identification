import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Identify A Stop Sign')

st.markdown("This web app identifies if a stop sign is in the image or not. Only two classification: stop sign or no stop sign")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                #st.pyplot(fig)


def predict(image, img_shape=224):
    classifier_model = "resnet_stop_model.h5"
    #IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    classifier_model = "resnet_stop_model.h5"
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    predictions = model.predict(test_image)
    class_names = np.array(['not stop', 'stop'])
    result = class_names[int(tf.round(predictions)[0][0])]
    results = f"{result} with a { (100 * np.max(predictions[0][0])).round(2) } % confidence." 
    #results = f"{result} with a { 100 * predictions[0][0] } % confidence."
    return results


      









    

if __name__ == "__main__":
    main()

