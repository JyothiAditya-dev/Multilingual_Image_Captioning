from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer

print("Loading translation models... Please wait.")

# Hindi translator (works fine)
translator_hi = pipeline(
    task="translation_en_to_hi",
    model="Helsinki-NLP/opus-mt-en-hi"
)

# Telugu translator (manual MarianMT)
model_name = "Helsinki-NLP/opus-mt-en-mul"

tokenizer_te = MarianTokenizer.from_pretrained(model_name)
model_te = MarianMTModel.from_pretrained(model_name)

print("Translation models loaded successfully.")


def translate(text, language):

    if language == "English":
        return text

    elif language == "Hindi":
        result = translator_hi(text)
        return result[0]['translation_text']

    elif language == "Telugu":

        # IMPORTANT: add Telugu language code
        telugu_text = ">>te<< " + text

        inputs = tokenizer_te(telugu_text, return_tensors="pt", padding=True)
        translated = model_te.generate(**inputs)

        output = tokenizer_te.decode(translated[0], skip_special_tokens=True)

        return output

    else:
        return text