import argparse
import pickle
import zipfile
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

import gdown
import numpy as np
from gdown.exceptions import FileURLRetrievalError
from PIL import Image
from tensorflow.keras.applications import efficientnet, inception_v3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_URL = "https://drive.google.com/uc?id=1gOvW_z0Nr-fX4bZt7Nb7xkvmfuqKxRmy"
MODEL_FILES = ("caption_model.h5", "tokenizer.pkl")


def ensure_model_files(model_dir=MODEL_DIR):
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if not missing:
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_dir) as temp:
        staging = Path(temp)
        archive = staging / "models.zip"
        try:
            if not gdown.download(MODEL_URL, str(archive), quiet=True):
                raise OSError("Google Drive did not return a file.")

            with zipfile.ZipFile(archive) as bundle:
                for name in missing:
                    entries = [entry for entry in bundle.namelist() if Path(entry).name == name]
                    if len(entries) != 1:
                        raise ValueError(f"The archive must contain exactly one {name}.")
                    with bundle.open(entries[0]) as source, (staging / name).open("wb") as target:
                        copyfileobj(source, target)
        except (OSError, ValueError, zipfile.BadZipFile, FileURLRetrievalError) as error:
            raise RuntimeError(
                f"Could not download the model bundle. Place caption_model.h5 and tokenizer.pkl in {model_dir}."
            ) from error

        for name in missing:
            (staging / name).replace(model_dir / name)


def load_caption_assets(model_dir=MODEL_DIR):
    ensure_model_files(model_dir)
    with (model_dir / "tokenizer.pkl").open("rb") as source:
        tokenizer = pickle.load(source)
    model = load_model(model_dir / "caption_model.h5", compile=False)
    return model, tokenizer


def load_feature_extractor(feature_size):
    if feature_size == 1280:
        base_model = efficientnet.EfficientNetB0(weights="imagenet")
        preprocess = efficientnet.preprocess_input
    elif feature_size == 2048:
        base_model = inception_v3.InceptionV3(weights="imagenet")
        preprocess = inception_v3.preprocess_input
    else:
        raise ValueError(f"Unsupported caption model feature size: {feature_size}")

    return Model(base_model.input, base_model.layers[-2].output), preprocess


def extract_features(image, model, preprocess):
    _, height, width, _ = model.input_shape
    image = image.convert("RGB").resize((width, height))
    batch = preprocess(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0))
    return model.predict(batch, verbose=0)


def generate_caption(model, tokenizer, features):
    max_length = model.inputs[1].shape[-1]
    words = ["startseq"]
    for _ in range(max_length):
        tokens = tokenizer.texts_to_sequences([" ".join(words)])
        sequence = pad_sequences(tokens, maxlen=max_length)
        prediction = model.predict([features, sequence], verbose=0)
        word = tokenizer.index_word.get(int(np.argmax(prediction[0])))
        if word is None or word == "endseq":
            break
        words.append(word)
    return " ".join(words[1:])


def main():
    parser = argparse.ArgumentParser(description="Generate an English caption for an image.")
    parser.add_argument("image", type=Path, help="Path to an image")
    args = parser.parse_args()

    try:
        with Image.open(args.image) as image:
            model, tokenizer = load_caption_assets()
            encoder, preprocess = load_feature_extractor(model.inputs[0].shape[-1])
            features = extract_features(image, encoder, preprocess)
            caption = generate_caption(model, tokenizer, features)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    print(caption)


if __name__ == "__main__":
    main()
