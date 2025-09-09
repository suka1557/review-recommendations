# BERT (Bidirectional Encoder Representation Transformers)

BERT has some requirments while feeding the data
* Use the tokenizer that is associated with that particular model
* Max Sequence Length allowed for BERT model = 512
* You need to make all sentences of same size using padding or truncating
* If there are multiple sentences being passed in 1 row, then you need to separate them with [SEP] token
* To pad, use [PAD] token
* The start of every sentence should be [CLS] token
* An Attention Mask needs to be passed along with the sentence, which is an array of 1's and 0's telling the model, which to the tokens are actual and which ones (padding ones) are added just for making sequence length same
* The [CLS] token needs to be at the start of the sequence once. It doesn't need to be added at the start of every sentence in the sequence.
* the embedding produced for the [CLS] token in the last layer is taken as encoded representation of the entire sequence
* For tasks with two sentences (e.g., sentence pair classification), BERT expects a token_type_ids array to distinguish between the first and second sentence (0 for the first, 1 for the second)

# Cased and Uncased version of BERT
## 1. Uncased BERT
* All text is lowercased before tokenization.
* Example: "Apple is better than apple." → "apple is better than apple."
* WordPiece vocabulary only contains lowercase tokens (apple, not Apple).
* Cannot distinguish between "Apple" (the company) and "apple" (the fruit).
* Model size is slightly smaller since fewer distinct tokens exist in the vocabulary.
* Good for tasks where casing is not important (e.g., sentiment analysis, topic classification).

## 2. Cased BERT
* Keeps the original casing (upper vs. lower) during tokenization.
* Example: "Apple is better than apple." → tokens include "Apple" and "apple".
* WordPiece vocabulary has separate entries for "Apple", "APPLE", "apple", etc.
* Helps the model capture case-sensitive nuances:
* "us" (pronoun) vs "US" (United States)
* "Apple" (company) vs "apple" (fruit)
* Vocabulary is larger than uncased.
* Better for tasks where case carries meaning (e.g., Named Entity Recognition, QA, information extraction).

## 3. Which to use?
<b>Use uncased if:</b>
Your dataset is noisy and casing is inconsistent.
You’re doing general classification where case doesn’t add much signal.

<b>Use cased if:</b>
Your task requires distinguishing entities or case-specific meanings.
You’re working with formal text where casing is reliable (news, legal docs, biomedical text).

👉 In practice:
<b>For NER, QA, relation extraction → cased usually performs better.</b>
<b>For sentiment classification, intent detection, topic classification → uncased is often enough.</b>

