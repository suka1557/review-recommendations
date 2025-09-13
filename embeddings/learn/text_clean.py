import re
import string
import nltk
from nltk.tokenize import sent_tokenize

# Make sure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class TextClean:
    def __init__(self):
        # Regex for emoji ranges
        self.emoji_pattern = re.compile(
            "["  
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002700-\U000027BF"  # dingbats
            u"\U0001F900-\U0001F9FF"  # supplemental symbols
            "]+",
            flags=re.UNICODE
        )
        self.punct_table = str.maketrans("", "", string.punctuation)

    def process_sentence(self, sentence: str):
        """Apply cleaning steps on a single sentence."""
        # Add start and end tokens
        sentence = f"<START> {sentence.strip()} <END>"

        # Extract emojis with positions (relative to this sentence)
        emojis = []
        for match in self.emoji_pattern.finditer(sentence):
            emojis.append([match.start(), match.end(), match.group()])

        return sentence, emojis

    def clean(self, text: str):
        """Process a full review with multiple sentences."""

        #Step 1: Lowercase Text
        text = text.lower()

        #Step 2: Break into Sentences
        sentences = sent_tokenize(text)

        # Step 3: Remove Punctuations
        sentences = [sent.translate(self.punct_table) for sent in sentences]

        # Step 4: Add start end tokens, extract emojis
        all_cleaned = []
        all_emojis = []

        for sent in sentences:
            cleaned, emojis = self.process_sentence(sent)
            all_cleaned.append(cleaned)
            all_emojis.extend(emojis)

        return all_cleaned, all_emojis


if __name__ == "__main__":
    cleaner = TextClean()

    text = "Wow!! The food was sooo good 😋🔥!! Service was excellent. But the waiter was rude 😡."
    cleaned_sentences, emojis = cleaner.clean(text)

    print("Cleaned sentences:", cleaned_sentences)
    print("Emojis:", emojis)