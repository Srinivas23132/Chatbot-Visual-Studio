import os
import json
import datetime
import csv
import ssl
import random
import nltk
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issue for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents.json file safely
try:
    file_path = os.path.join(os.path.dirname(__file__), "intents.json")
    with open(file_path, "r", encoding="utf-8") as file:
        intents = json.load(file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    st.error(f"Error loading intents.json: {e}")
    st.stop()

# Create TF-IDF vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess data for training
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Ensure chat log exists
chat_log_path = "chat_log.csv"
if not os.path.exists(chat_log_path):
    with open(chat_log_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

# Streamlit app
def main():
    st.title("Chatbot using NLP & Logistic Regression")

    # Sidebar Menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Page
    if choice == "Home":
        st.write("Welcome! Type a message and press Enter to chat.")

        # Maintain a persistent counter for Streamlit inputs
        if "chat_counter" not in st.session_state:
            st.session_state.chat_counter = 0
        st.session_state.chat_counter += 1

        # User input box
        user_input = st.text_input("You:", key=f"user_input_{st.session_state.chat_counter}")

        if user_input.strip():  # Prevent empty messages
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=100, key=f"chatbot_response_{st.session_state.chat_counter}")

            # Log chat in CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(chat_log_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Stop chatbot on exit command
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a great day! 😊")
                st.stop()

    # Conversation History Page
    elif choice == "Conversation History":
        st.header("📜 Conversation History")

        if os.path.exists(chat_log_path):
            df = pd.read_csv(chat_log_path)
            with st.expander("Click to view full conversation"):
                st.dataframe(df, width=800, height=400)
        else:
            st.write("No conversation history found.")

    # About Page
    elif choice == "About":
        st.write("### 🤖 About This Chatbot")
        st.write("""
        - This chatbot is trained using **Natural Language Processing (NLP)** and **Logistic Regression**.
        - Uses **TF-IDF Vectorization** for text representation.
        - Built with **Streamlit** for an interactive UI.
        - Stores conversation history for reference.
        """)

        st.subheader("🔹 Project Overview:")
        st.write("""
        1. **NLP & Machine Learning**: The chatbot is trained on labeled intents using NLP techniques.
        2. **Streamlit Web Interface**: Provides an interactive UI for chatting.
        """)

        st.subheader("🔹 How It Works:")
        st.write("""
        - User inputs a message.
        - The chatbot predicts the intent using Logistic Regression.
        - Selects a random response from the matching intent.
        - Saves the conversation in a CSV file.
        """)

        st.subheader("🔹 Future Improvements:")
        st.write("""
        - Use **Deep Learning (LSTMs or Transformers)** for better accuracy.
        - Add **Context Awareness** to maintain conversation flow.
        """)

if __name__ == '__main__':
    main()
