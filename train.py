import os
import pickle
import numpy as np
import re

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)

from tensorflow.keras.models import Model


# =====================================
# PATHS
# =====================================

CAPTION_PATH = "dataset/Flickr8k.token.txt"
FEATURE_PATH = "features/features.pkl"

MODEL_DIR = "models"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "caption_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================
# PARAMETERS (CPU optimized)
# =====================================

EMBED_DIM = 256
FF_DIM = 512
NUM_HEADS = 2

MAX_LENGTH = 34
VOCAB_SIZE = 8000

BATCH_SIZE = 32
EPOCHS = 10


# =====================================
# LOAD FEATURES
# =====================================

print("Loading features...")

with open(FEATURE_PATH, "rb") as f:
    features = pickle.load(f)

print("Features loaded:", len(features))


# =====================================
# LOAD CAPTIONS (100% CORRECT FIX)
# =====================================

def load_captions(filename):

    captions = {}

    with open(filename, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if len(line) == 0:
                continue

            # Split only first space
            split_index = line.find(" ")

            if split_index == -1:
                continue

            image_part = line[:split_index]
            caption = line[split_index + 1:]

            # Remove #0 and .jpg
            img_id = image_part.split("#")[0]
            img_id = os.path.splitext(img_id)[0]

            # Clean caption
            caption = caption.lower()
            caption = re.sub(r"[^a-z\s]", "", caption)
            caption = caption.strip()

            caption = "startseq " + caption + " endseq"

            if img_id not in captions:
                captions[img_id] = []

            captions[img_id].append(caption)

    return captions


print("Loading captions...")

captions_dict = load_captions(CAPTION_PATH)

print("Captions loaded:", len(captions_dict))


# =====================================
# TOKENIZER
# =====================================

print("Preparing tokenizer...")

all_captions = []

for key in captions_dict:
    all_captions.extend(captions_dict[key])

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<unk>"
)

tokenizer.fit_on_texts(all_captions)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved")


# =====================================
# CREATE TRAINING DATA
# =====================================

print("Preparing training sequences...")

X1 = []
X2 = []
y = []

for img_id, captions in captions_dict.items():

    if img_id not in features:
        continue

    feature = features[img_id]

    for caption in captions:

        seq = tokenizer.texts_to_sequences([caption])[0]

        for i in range(1, len(seq)):

            in_seq = seq[:i]
            out_seq = seq[i]

            in_seq = pad_sequences(
                [in_seq],
                maxlen=MAX_LENGTH
            )[0]

            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)


X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

print("Training samples:", len(X1))


# =====================================
# TRANSFORMER DECODER
# =====================================

def transformer_block(x, encoder_output):

    encoder_output = tf.expand_dims(encoder_output, axis=1)

    attention1 = MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=EMBED_DIM
    )(x, x)

    x = LayerNormalization()(x + attention1)

    attention2 = MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=EMBED_DIM
    )(x, encoder_output)

    x = LayerNormalization()(x + attention2)

    ff = Dense(FF_DIM, activation="relu")(x)
    ff = Dense(EMBED_DIM)(ff)

    x = LayerNormalization()(x + ff)

    return x


# =====================================
# BUILD MODEL
# =====================================

print("Building model...")

image_input = Input(shape=(1280,))

image_dense = Dense(EMBED_DIM)(image_input)

caption_input = Input(shape=(MAX_LENGTH,))

caption_embed = Embedding(
    VOCAB_SIZE,
    EMBED_DIM
)(caption_input)

decoder_output = transformer_block(
    caption_embed,
    image_dense
)

decoder_output = GlobalAveragePooling1D()(decoder_output)

outputs = Dense(
    VOCAB_SIZE,
    activation="softmax"
)(decoder_output)

model = Model(
    inputs=[image_input, caption_input],
    outputs=outputs
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam"
)

model.summary()


# =====================================
# TRAIN MODEL
# =====================================

print("Starting training...")

model.fit(
    [X1, X2],
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)


# =====================================
# SAVE MODEL
# =====================================

model.save(MODEL_PATH)

print("Training completed successfully.")