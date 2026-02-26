import os
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

CAPTION_FILE = "dataset/Flickr8k.token.txt"
TOKENIZER_FILE = "models/tokenizer.pkl"
METADATA_FILE = "models/text_metadata.pkl"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_captions(filename):
    captions = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            # Handle both TAB and SPACE formats
            if '\t' in line:
                img_part, caption = line.split('\t')
            else:
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    continue
                img_part, caption = parts
            
            img_id = img_part.split('#')[0]
            
            caption = clean_text(caption)
            caption = "startseq " + caption + " endseq"
            
            if img_id not in captions:
                captions[img_id] = []
                
            captions[img_id].append(caption)
    
    return captions

print("Loading captions...")

if not os.path.exists(CAPTION_FILE):
    print("ERROR: Caption file not found!")
    exit()

captions_dict = load_captions(CAPTION_FILE)

all_captions = []
for key in captions_dict:
    all_captions.extend(captions_dict[key])

print("Total images:", len(captions_dict))
print("Total captions:", len(all_captions))

if len(all_captions) == 0:
    print("ERROR: No captions loaded. Check file format.")
    exit()

print("Creating tokenizer...")

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

print("Vocabulary size:", vocab_size)
print("Max caption length:", max_length)

os.makedirs("models", exist_ok=True)

pickle.dump(tokenizer, open(TOKENIZER_FILE, "wb"))
pickle.dump(
    {"vocab_size": vocab_size, "max_length": max_length},
    open(METADATA_FILE, "wb")
)

print("\nTokenizer saved successfully!")
print("Metadata saved successfully!")
print("Text processing completed ✅")