import streamlit as st
from PIL import Image
from transformers import pipeline

from inference import load_captioner


@st.cache_resource
def get_captioner():
    return load_captioner()


@st.cache_resource
def load_translators():
    hindi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", framework="pt")
    telugu = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul", framework="pt")
    return hindi, telugu


def main():
    st.title("Multilingual Image Captioning System")
    st.write("Upload an image to generate captions in English, Hindi, and Telugu.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Generating the English caption…"):
            english = get_captioner()(image)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not generate a caption. {error}")
        return

    if not english:
        st.warning("The model did not generate a caption for this image.")
        return

    st.subheader("English Caption")
    st.write(english)

    try:
        with st.spinner("Translating the caption…"):
            hindi, telugu = load_translators()
            hindi_caption = hindi(english)[0]["translation_text"]
            telugu_caption = telugu(f">>tel<< {english}")[0]["translation_text"]
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Translation is unavailable. {error}")
        return

    st.subheader("Hindi Caption")
    st.write(hindi_caption)
    st.subheader("Telugu Caption")
    st.write(telugu_caption)


if __name__ == "__main__":
    main()
