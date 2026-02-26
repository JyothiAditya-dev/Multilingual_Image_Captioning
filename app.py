import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import gdown
import zipfile
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from transformers import pipeline

# =========================
# Download models from Google Drive
# =========================

def download_models():

    if not os.path.exists("models"):
        os.makedirs("models")

    # change file id if needed
    file_id = "1gOvW_z0Nr-fX4bZt7Nb7xkvmfuqKxRmy"
    zip_path = "models/models.zip"

    if not os.path.exists("models/tokenizer.pkl"):

        st.write("Downloading model files...")

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("models")

        st.write("Models downloaded successfully")


download_models()

# =========================
# Load tokenizer
# =========================

@st.cache_resource
def load_tokenizer():

    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


# =========================
# Load caption model
# =========================

@st.cache_resource
def load_caption_model():

    model = tf.keras.models.load_model("models/caption_model.h5")

    return model


# =========================
# Load feature extractor
# =========================

@st.cache_resource
def load_feature_extractor():

    base_model = InceptionV3(weights="imagenet")
    model = Model(base_model.input, base_model.layers[-2].output)

    return model


# =========================
# Translation models
# =========================

@st.cache_resource
def load_translators():

    translator_hi = pipeline(
        "translation_en_to_hi",
        model="Helsinki-NLP/opus-mt-en-hi"
    )

    translator_te = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-mul"
    )

    return translator_hi, translator_te


# =========================
# Caption generation function
# =========================

def generate_caption(model, tokenizer, photo, max_length=34):

    in_text = "startseq"

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    final = in_text.replace("startseq", "").replace("endseq", "")

    return final.strip()


# =========================
# Extract image features
# =========================

def extract_features(image, model):

    image = image.resize((299, 299))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)

    return feature


# =========================
# Translate function
# =========================

def translate(text, translator_hi, translator_te):

    hindi = translator_hi(text)[0]["translation_text"]

    telugu = translator_te(text)[0]["translation_text"]

    return hindi, telugu


# =========================
# Streamlit UI
# =========================

st.title("Multilingual Image Captioning System")

st.write("Upload an image to generate captions in English, Hindi, and Telugu")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    tokenizer = load_tokenizer()
    caption_model = load_caption_model()
    feature_model = load_feature_extractor()
    translator_hi, translator_te = load_translators()

    feature = extract_features(image, feature_model)

    english_caption = generate_caption(caption_model, tokenizer, feature)

    hindi_caption, telugu_caption = translate(
        english_caption,
        translator_hi,
        translator_te
    )

    st.subheader("English Caption:")
    st.write(english_caption)

    st.subheader("Hindi Caption:")
    st.write(hindi_caption)

    st.subheader("Telugu Caption:")
    st.write(telugu_caption)
