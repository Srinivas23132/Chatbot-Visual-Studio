import json
import nltk
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the intents file (Make sure the path is correct)
file_path = r"C:\Users\bhava\Downloads\Chatbot-Implementation-using-Python-NLP-main\Chatbot-Implementation-using-Python-NLP-main\intents.json"

with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()

# Prepare the data
tags = []
patterns = []
responses = []

for intent in intents:
    if isinstance(intent, dict):  # Ensure it's a dictionary
        for pattern in intent['patterns']:
            # Tokenize each word in the pattern
            word_list = nltk.word_tokenize(pattern)
            patterns.append(pattern)
            tags.append(intent['tag'])
            responses.append(intent['responses'])

# Prepare training data
X = []
y = []

# Tokenize and lemmatize each word in the patterns
for pattern in patterns:
    word_list = nltk.word_tokenize(pattern)
    X.append([lemmatizer.lemmatize(w.lower()) for w in word_list])

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(tags)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a bag of words
all_words = [lemmatizer.lemmatize(w.lower()) for pattern in patterns for w in nltk.word_tokenize(pattern)]
all_words = sorted(list(set(all_words)))

# Create training data
training_sentences = []
training_labels = []

for doc, tag in zip(X_train, y_train):
    training_sentences.append(doc)
    training_labels.append(tag)

# Convert to numpy arrays
X_train = np.array(training_sentences)
y_train = np.array(training_labels)

# Create a neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(all_words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(tags)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("chatbot_model.h5")

# Function to predict the category of a message
def predict_class(msg):
    p = nltk.word_tokenize(msg)
    p = [lemmatizer.lemmatize(w.lower()) for w in p]
    bow = [0] * len(all_words)

    for i, word in enumerate(all_words):
        if word in p:
            bow[i] = 1

    return model.predict(np.array([bow]))[0]

# Function to get a response based on the predicted class
def get_response(msg):
    predicted_class = np.argmax(predict_class(msg))
    response = random.choice(responses[predicted_class])
    return response

# Streamlit app code
st.title("Chatbot")

st.markdown("""
    This is a chatbot that uses natural language processing (NLP) to understand and respond to user inputs. 
    You can ask it questions or just chat with it. Try greeting the bot or asking it questions!
""")

user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.write("Chatbot: " + response)
