import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sentence_transformers import SentenceTransformer
import keras
model = keras.models.load_model('model.h5')

ip = {'i want to die', 'this world is so fucking shit', 'i wish i was dead'}

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Define a function to preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join the tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

labeldict ={'Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down': 0,
            'Feeling-down-depressed-or-hopeless': 1, 
            'Feeling-tired-or-having-little-energy': 2, 
            'Little-interest-or-pleasure-in-doing ': 3,
            'Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual':4, 
            'Poor-appetite-or-overeating': 5, 
            'Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way': 6, 
            'Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television': 7, 
            'Trouble-falling-or-staying-asleep-or-sleeping-too-much': 8}
embedder = SentenceTransformer('google-bert/bert-base-uncased')
embedder.max_seq_length = 512
results = {}
for input_string in ip:
    # Preprocess the input string
    processed_input = preprocess_text(input_string)

    # Encode the processed input using the SentenceTransformer model
    embedded_input = embedder.encode([processed_input], show_progress_bar=False)

    # Make predictions for the input
    input_predictions = model.predict(embedded_input)
    input_predictions = pd.DataFrame(input_predictions)
    input_predictions = input_predictions.apply(lambda x: [1 if val >= 0.349802 else 0 for val in x])

    # Get the labels for input predictions
    input_predicted_labels = [(label, "yes" if input_predictions[value][0] == 1 else "no") for label, value in labeldict.items()]

    # Store the result in the dictionary
    results[input_string] = input_predicted_labels

# Ensure that all variables and functions are used correctly

# Store the results in a new file called results.txt
with open('results.txt', 'w') as file:
    file.write(str(results))

print("Results stored in results.txt")