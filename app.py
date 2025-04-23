import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import os

# Fix the path separator
model_path = os.path.join('pickle', 'simple_rnn_best_model.h5')


word_to_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_to_index.items()}


model = load_model(model_path)


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        idx = word_to_index.get(word, 2) + 3
        if idx >= 10000:
            idx = 9999
        encoded_review.append(idx)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    pred = model.predict(preprocessed_input)

    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'

    return sentiment, pred[0][0]


st.title("IMDB Movie Review Sentiment Analysis")
st.subheader(
    "This app uses a pre-trained LSTM model to predict the sentiment of movie reviews.")
st.divider()
st.write("Enter a movie review to classify the sentiment as positive or negative")

user_input = st.text_area("Enter your review here:", max_chars=500)

if st.button("Predict Sentiment"):
    if user_input:
        sent, predi = predict_sentiment(user_input)
        st.write(f"Sentiment: {sent}")
        st.write(f"Prediction score: {predi:.2f}")
    else:
        st.write("Please enter a review to predict sentiment.")
