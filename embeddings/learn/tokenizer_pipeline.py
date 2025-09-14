import os
import sys
import pandas as pd

from embeddings.learn.text_clean import TextClean
from embeddings.learn.train_tokenizer import ReviewTokenizer

#Define Paths
PROJECT_ROOT = os.path.join("/home/ar-in-u-301/Documents/codes", "review-recommendations")

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "data", "artifacts")

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

#Get Train Data
train = pd.read_feather(os.path.join(DATA_PATH, "train_review.feather"))

def write_sentences_to_file(sentences, filepath):
    """Write a list of sentences to a text file, one per line. Append if file exists."""
    mode = 'a' if os.path.exists(filepath) else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

#Save Cleaned sentences in a txt file
review_text_file = os.path.join(DATA_PATH, "train_review_text.txt")
text_cleaner = TextClean()
for i,review_text in enumerate(train['text']):
    sentences,emojis = text_cleaner.clean(text=review_text)

    #Save to sentences
    write_sentences_to_file(sentences=sentences, filepath=review_text_file)

    if i > 0 and i%1000 == 0:
        print(f"Processed {i+1} files, Remaining: {len(train) - (i+1)}")

#Train tokenizer using the review text file
VOCAB_SIZE = 30000
SPECIAL_TOKENS = ['<START>', '<END>', '<PAD>', '<UNK>']


#initialise
review_tokenizer = ReviewTokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
#train
review_tokenizer.train(file_path=review_text_file)

TOEKNIZER_FOLDER = os.path.join(ARTIFACTS_PATH, "tokenizers", "v1")
os.makedirs(TOEKNIZER_FOLDER, exist_ok=True)
#save
review_tokenizer.save_tokenizer(save_dir=TOEKNIZER_FOLDER)

