from transformers import pipeline

print("Loading translation models... Please wait.")

# Hindi translator
translator_hi = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-hi"
)

# Multilingual translator (supports Telugu)
translator_multi = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-mul"
)

print("Translation models loaded successfully.")

def translate(text, language):
    
    if language == "Hindi":
        result = translator_hi(text)
        return result[0]['translation_text']
    
    elif language == "Telugu":
        result = translator_multi(text, tgt_lang="te")
        return result[0]['translation_text']
    
    else:
        return text
