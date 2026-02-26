from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading translation models... Please wait.")

# Hindi model
model_hi_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer_hi = AutoTokenizer.from_pretrained(model_hi_name)
model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_hi_name)

# Multilingual model for Telugu
model_te_name = "Helsinki-NLP/opus-mt-en-mul"
tokenizer_te = AutoTokenizer.from_pretrained(model_te_name)
model_te = AutoModelForSeq2SeqLM.from_pretrained(model_te_name)

print("Translation models loaded successfully.")


def translate(text, language):

    if language == "Hindi":

        inputs = tokenizer_hi(text, return_tensors="pt", padding=True)
        outputs = model_hi.generate(**inputs)
        translated = tokenizer_hi.decode(outputs[0], skip_special_tokens=True)
        return translated


    elif language == "Telugu":

        inputs = tokenizer_te(text, return_tensors="pt", padding=True)
        outputs = model_te.generate(**inputs)
        translated = tokenizer_te.decode(outputs[0], skip_special_tokens=True)
        return translated


    else:
        return text
