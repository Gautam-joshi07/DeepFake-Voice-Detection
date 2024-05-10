


import streamlit as st
import pandas as pd
import numpy as np
import librosa
import os
from PIL import Image
from io import BytesIO
import tensorflow as tf
from st_audiorec import st_audiorec
import altair
import keras
import librosa.display
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array

os.environ["KERAS_BACKEND"] = "tensorflow"

st.set_page_config(page_title="Deepfake Audio")
class_names = ['real', 'fake']


def file_save(file_sound):
    with open(os.path.join('audio_files/', file_sound.name), 'wb') as f:
        f.write(file_sound.getbuffer())

    return file_sound.name


def create_spec(sound):
    audio_file = os.path.join('audio_files/', sound)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y, sr = librosa.load(audio_file)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig('mel_spectrogram.png')
    image_data = load_img('mel_spectrogram.png', target_size=(224, 224))
    st.image(image_data)

    return image_data


def pred(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255
    img_batch = np.expand_dims(img_array1, axis=0)

    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)

    return class_label, prediction


def file_upload_page():
    st.write("## File Upload Page")
    uploaded_file = st.file_uploader('Upload a .wav or .mp3 file', type=['wav', 'mp3'])
    if uploaded_file is not None:
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        st.write('### Spectrogram Image:')
        file_save(uploaded_file)
        sound = uploaded_file.name
        with st.spinner('Fetching Results...'):
            spec = create_spec(sound)
            model = tf.keras.models.load_model('model/model.keras')
        st.write('### Classification results:')
        class_label, prediction = pred(spec, model)
        st.write("#### The uploaded audio file is " + class_names[class_label])


def record_audio_page():

    st.write("### Record Your Voice")
    st.write("- ** After that it will automatically process and gives results that audio file is real or fake(AI generated)")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        st.write("### Spectrogram Image:")
        # Save the recorded audio as a file
        with open('audio_files/recorded_audio.wav', 'wb') as f:
            f.write(wav_audio_data)
        sound = 'recorded_audio.wav'
        with st.spinner('Fetching Results...'):
            spec = create_spec(sound)
            model = tf.keras.models.load_model('model/model.keras')
        st.write('### Classification results:')
        class_label, prediction = pred(spec, model)
        st.write("#### The recorded audio is " + class_names[class_label])


def main():
    # Default page


    # Sidebar to switch between pages
    page_options = ['Information', 'Upload Audio File', 'Record Audio']
    selected_page = st.sidebar.selectbox('Select Page', page_options)

    # Show corresponding page based on selection
    if selected_page == 'Information':
        show_information_page()
    elif selected_page == 'Upload Audio File':
        file_upload_page()
    elif selected_page == 'Record Audio':
        record_audio_page()


def show_information_page():
    st.write("## Deepfake Audio Classification")
    st.write("This web app allows you to classify audio files as real or fake.")
    st.write("Please select an option from the dropdown menu to proceed.")

    st.write("## Information Page")
    st.write("This page provides information about the Deepfake Audio Classification web app.")

    st.write("## Audio Features")
    st.write("- **Spectrogram:** A visual representation of the audio frequency content.")
    st.write("- **Classification results:** The prediction of whether the audio is real or fake.")
    st.write("- **Model:** Deep learning model trained to classify audio files.")



if __name__ == "__main__":
    main()
