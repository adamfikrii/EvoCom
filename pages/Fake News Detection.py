import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
model = load_model('fake_news_rnn_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define max length (should match training setup)
max_length = 500

# Streamlit App
st.title("Fake News Detection App 📰")
st.write("Input a news article to check if it's real or fake.")

# User input
news_input = st.text_area("Enter the news article text:")

if st.button("Predict"):
    if news_input.strip():
        # Preprocess the input
        sequence = tokenizer.texts_to_sequences([news_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Predict
        prediction = model.predict(padded_sequence)[0][0]
        label = "Real" if prediction > 0.5 else "Fake"

        # Display results
        st.subheader("Prediction:")
        st.write(f"The article is **{label}**.")
        st.write(f"Confidence: {prediction * 100:.2f}%")
    else:
        st.error("Please enter a valid news article.")
