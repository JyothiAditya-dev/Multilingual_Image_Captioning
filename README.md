# Multilingual Image Captioning using Deep Learning

## Project Description
This project generates captions for images using a trained deep learning model and translates the captions into Hindi and Telugu.

It uses:
- CNN + LSTM for caption generation
- HuggingFace translation models for Hindi and Telugu
- Streamlit for user interface

---

## Features
• Generate captions from images  
• Translate captions into Hindi  
• Translate captions into Telugu  
• Deep learning trained model  
• Easy-to-use Streamlit interface  

---

## Project Structure

Multilingual_Image_Captioning/

dataset/ → Flickr8k dataset  
features/ → extracted image features  
models/ → trained model (download separately)  
inference.py → caption generation  
translator.py → translation code  
test_translate.py → translation testing  
train.py → model training  
extract_features.py → feature extraction  

---

## Model Download

Download trained model from Google Drive:

[Download the model archive](https://drive.google.com/file/d/1gOvW_z0Nr-fX4bZt7Nb7xkvmfuqKxRmy/view)

After download, place file in:

`models/caption_model.h5` and `models/tokenizer.pkl`

---

## Installation

Install requirements:

`python -m pip install -r requirements.txt`

---

## Run Caption Generator

python inference.py

---

## Run Translation Test

python test_translate.py

---

## Run Streamlit Web App

streamlit run app.py

---

## Technologies Used

Python  
TensorFlow  
PyTorch  
Transformers  
Streamlit  
CNN  
LSTM  

---

## Author

Jyothi Aditya
