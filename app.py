import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import gdown
import zipfile
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import efficientnet, inception_v3
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

    if not all(os.path.isfile(f"models/{name}") for name in ("tokenizer.pkl", "caption_model.h5")):

        st.write("Downloading model files...")

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("models")

        st.write("Models downloaded successfully")



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

    model = tf.keras.models.load_model("models/caption_model.h5", compile=False)

    return model


# =========================
# Load feature extractor
# =========================

@st.cache_resource
def load_feature_extractor(feature_size):
    if feature_size == 1280:
        encoder = efficientnet
        base_model = encoder.EfficientNetB0(weights="imagenet")
    elif feature_size == 2048:
        encoder = inception_v3
        base_model = encoder.InceptionV3(weights="imagenet")
    else:
        raise ValueError(f"Unsupported caption model feature size: {feature_size}")

    return Model(base_model.input, base_model.layers[-2].output), encoder.preprocess_input


# =========================
# Translation models
# =========================

@st.cache_resource
def load_translators():

    translator_hi = pipeline(
        "translation",
        framework="pt",
        model="Helsinki-NLP/opus-mt-en-hi"
    )

    translator_te = pipeline(
        "translation",
        framework="pt",
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

def extract_features(image, model, preprocess_input):
    _, height, width, _ = model.input_shape
    image = image.convert("RGB").resize((width, height))
    image = preprocess_input(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0))
    feature = model.predict(image, verbose=0)

    return feature


# =========================
# Translate function
# =========================

def translate(text, translator_hi, translator_te):

    hindi = translator_hi(text)[0]["translation_text"]

    telugu = translator_te(f">>tel<< {text}")[0]["translation_text"]

    return hindi, telugu


# =========================
# Streamlit UI
# =========================

st.title("Multilingual Image Captioning System")

st.write("Upload an image to generate captions in English, Hindi, and Telugu")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:

    download_models()
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    tokenizer = load_tokenizer()
    caption_model = load_caption_model()
    feature_model, preprocess_input = load_feature_extractor(caption_model.inputs[0].shape[-1])
    translator_hi, translator_te = load_translators()

    feature = extract_features(image, feature_model, preprocess_input)

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
