import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def mobilenetv2_imagenet():
    st.title("image classification with MobileNetV2")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('classifying...')

        # Load the MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')

        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Predict the image
        prediction = model.predict(img_array)
        result = tf.keras.applications.mobilenet_v2.decode_predictions(prediction)

        for i, (imagenet_id, label, score) in enumerate(result[0]):
            st.write(f"{label} {score:.2f}%")

def cifar10_classification():
    st.title("Image classification with CIFAR-10")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write('classifying...')

        # Load the CIFAR-10 model
        model = tf.keras.models.load_model('cnn_model.h5')
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array =img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the image
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence =  np.max(prediction)

        st.write(f"Predicted class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}%")

def main():
    st.sidebar.title('navigation')
    choice = st.sidebar.selectbox("Model", ('MobileNetV2', 'CIFAR-10'))

    if choice == 'MobileNetV2':
        mobilenetv2_imagenet()
    elif choice == 'CIFAR-10':
        cifar10_classification()

if __name__ == '__main__':
    main()