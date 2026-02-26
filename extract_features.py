import os
import numpy as np
import pickle
from tqdm import tqdm

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Paths
IMAGE_PATH = "dataset/Images"
FEATURE_PATH = "features/features.pkl"

# Load EfficientNetB0 model
base_model = EfficientNetB0(weights="imagenet")

model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

print("EfficientNetB0 loaded successfully")

# Extract features
features = {}

for img_name in tqdm(os.listdir(IMAGE_PATH)):

    img_path = os.path.join(IMAGE_PATH, img_name)

    img = load_img(img_path, target_size=(224,224))

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    feature = model.predict(img, verbose=0)

    img_id = img_name.split(".")[0]

    features[img_id] = feature.flatten()

# Save features
with open(FEATURE_PATH, "wb") as f:
    pickle.dump(features, f)

print("Feature extraction completed and saved.")