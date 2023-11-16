from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import re
import nltk
import string
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import pickle

app = Flask(__name__)

# Load your trained SVM model
with open('svm_model.pkl', 'rb') as model_file:
    support = pickle.load(model_file)

# Load your TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    predicted_label = None
    input_text = request.form.get('input_text', '')  # Set default value to an empty string

    if request.method == 'POST':
        predicted_label = classify_text(input_text)

    return render_template('classification.html', input_text=input_text, predicted_label=predicted_label)

def preprocess_input_text(input_text):
    stemmer = PorterStemmer()

    extended_stopwords = nltk.corpus.stopwords.words("english")
    other_exclusions = ["#ff", "ff", "rt"]
    extended_stopwords.extend(other_exclusions)

    tweet_space = re.sub(r'\s+', ' ', input_text)
    tweet_name = re.sub(r'@[\w\-]+', '', tweet_space)
    tweet_no_links = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet_name)
    tweet_no_punctuation = re.sub("[^a-zA-Z]", " ", tweet_no_links)
    tweet_stripped = tweet_no_punctuation.strip()
    tweet_no_numbers = re.sub(r'\d+(\.\d+)?', 'numbr', tweet_stripped)
    tweet_lower = tweet_no_numbers.lower()
    tokenized_tweet = tweet_lower.split()
    tokenized_tweet = [stemmer.stem(token) for token in tokenized_tweet if token not in extended_stopwords]
    processed_input_text = ' '.join(tokenized_tweet)
    return processed_input_text

def classify_text(input_text):
    processed_input_text = preprocess_input_text(input_text)
    tfidf_sample = tfidf_vectorizer.transform([processed_input_text])
    predicted_label = support.predict(tfidf_sample)[0]
    class_labels = {
        0: "Hate Speech",
        1: "Offensive",
        2: "Neither"
    }
    predicted_label_text = class_labels[predicted_label]
    return predicted_label_text

if __name__ == '__main__':
    app.run(debug=True)
