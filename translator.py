from transformers import pipeline

print("Loading translation models... Please wait.")

# Hindi translator
translator_hi = pipeline(
    task="translation_en_to_hi",
    model="Helsinki-NLP/opus-mt-en-hi"
)

# Telugu translator using multilingual model
translator_te = pipeline(
    task="translation_en_to_te",
    model="Helsinki-NLP/opus-mt-en-mul"
)

print("Translation models loaded successfully.")


def translate(text, language):

    if language == "Hindi":
        result = translator_hi(text)
        return result[0]['translation_text']

    elif language == "Telugu":
        result = translator_te(text)
        return result[0]['translation_text']

    else:
        return text
