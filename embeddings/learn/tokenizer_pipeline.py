import os
import pandas as pd
import yaml
from typing import Dict

from embeddings.learn.text_clean import TextClean
from embeddings.learn.tokenizer import ReviewTokenizer

#Define Paths
PROJECT_ROOT = os.path.join("/home/ar-in-u-301/Documents/codes", "review-recommendations")
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

#Read COnfigs
with open(CONFIG_FILE_PATH, 'r') as f:
    yaml_configs = yaml.safe_load(f)

#Print Logs
def print_logs(config_dict: Dict):
    for key, value in config_dict.items():
        if isinstance(value, str) or isinstance(value, float) or isinstance(value, int):
            print(key, ":", value)
        elif isinstance(value, list):
            print()
            print(f"Items in List - {key}: ")
            for item in value:
                print(item)
            print(f"Completed parsing List: {key}")
        elif isinstance(value, dict):
            print()
            print(f"Items in Dict - {key}: ")
            print_logs(value)
            print(f"Completed Parsing Dict: {key} ")

# Parsing Logs
print_logs(yaml_configs)


#Create Required folders if it does not exits
DATA_PATH = os.path.join(PROJECT_ROOT, yaml_configs.get("data_folder", "data"))
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, yaml_configs.get("artifacts_folder", "data/artifacts"))

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

#Get Train Data
train = pd.read_feather(os.path.join(DATA_PATH, yaml_configs.get("data_files", {}).get("train_file", "train_review.feather")) )

def write_sentences_to_file(sentences, filepath):
    """Write a list of sentences to a text file, one per line. Append if file exists."""
    mode = 'a' if os.path.exists(filepath) else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

#Save Cleaned sentences in a txt file
review_text_file = os.path.join(DATA_PATH, yaml_configs.get("data_files", {}).get("review_text_sentences", "train_review_sentences.txt"))
text_cleaner = TextClean()
for i,review_text in enumerate(train['text']):
    sentences,emojis = text_cleaner.clean(text=review_text)

    #Save to sentences
    write_sentences_to_file(sentences=sentences, filepath=review_text_file)

    if i > 0 and i%1000 == 0:
        print(f"Processed {i+1} files, Remaining: {len(train) - (i+1)}")

#Train tokenizer using the review text file
VOCAB_SIZE = yaml_configs.get("tokenizer", {}).get("vocab_size", 30000)
SPECIAL_TOKENS = yaml_configs.get("tokenizer", {}).get("special_tokens", [])


#initialise
review_tokenizer = ReviewTokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
#train
review_tokenizer.train(file_path=review_text_file)

TOEKNIZER_FOLDER = os.path.join(ARTIFACTS_PATH, yaml_configs.get("tokenizer", {}).get("folder", "tokenizers/v1"))
os.makedirs(TOEKNIZER_FOLDER, exist_ok=True)
#save
review_tokenizer.save_tokenizer(save_dir=TOEKNIZER_FOLDER)

