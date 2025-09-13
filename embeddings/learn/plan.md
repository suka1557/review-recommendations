We have the review text data and the ratings given to the user

We'll use a supervised approach, wehere we'll train a small neural network to learn embeddings
with a goal to map the information from review to final ratings.

This is in part inspired by Tang et al. (2014) — “Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification.”

# Plan for Pipeline:

## Step 1: Clean Text
    * Lowercase the Text
    * Extact Sentences from the Text using NLTK Sentence Tokenizers
    * Extract Emoji's with their position in text and use this as extra tokens to learn embeddings for

## Step 2: Tokenization
    * Let's go with WordPiece tokenizer

## Step 3: Starting Embeddings
    * Option 1: Start with Random <D> dimensional vectors - Learnable

## Step 4: Create a Feed Forward Neural Network Architecture to Predict Ratings
    * EMbeddings will be learned while training this

    * Output Labels: We have ratings between 0 -5
    * Train 2 Types of Models:
        * Output is classes: Cross Entroy Loss
        * Output is numerical: Ordinal Regression

## Step 5: Compare the embeddings learned with 
    * Sentence Transformer Embeddings (384 dimensional)
