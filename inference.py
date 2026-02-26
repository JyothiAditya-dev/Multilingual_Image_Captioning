import os
import pickle
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Paths
MODEL_PATH = "models/caption_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

IMAGE_PATH = "test.jpg"   # change to your test image


# Parameters
MAX_LENGTH = 34
EMBED_DIM = 256


# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# Load model
model = load_model(MODEL_PATH)


# Load EfficientNet encoder
base_model = EfficientNetB0(weights="imagenet")

encoder = tf.keras.Model(
    base_model.input,
    base_model.layers[-2].output
)


# Extract features from image
def extract_feature(image_path):

    image = load_img(image_path, target_size=(224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = encoder.predict(image, verbose=0)

    return feature


# Convert index to word
def idx_to_word(index, tokenizer):

    for word, idx in tokenizer.word_index.items():

        if idx == index:

            return word

    return None


# Generate caption
def generate_caption(model, tokenizer, feature):

    in_text = "startseq"

    for i in range(MAX_LENGTH):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

        yhat = model.predict([feature, sequence], verbose=0)

        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "")

    return caption.strip()


# Run inference
feature = extract_feature(IMAGE_PATH)

caption = generate_caption(model, tokenizer, feature)

print("\nGenerated Caption:")
print(caption)