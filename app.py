import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import pickle

# ---------- NLTK SETUP ----------
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except:
    pass

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------- LOAD DATA ----------
file_path = os.path.join(os.path.dirname(__file__), "intents.json")
with open(file_path) as file:
    intents = json.load(file)

# ---------- LOAD PRETRAINED MODEL ----------
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    clf = pickle.load(open("model.pkl", "rb"))
    return vectorizer, clf

vectorizer, clf = load_model()

# ---------- CHATBOT ----------
def chatbot(input_text):

    input_vector = vectorizer.transform([input_text])

    probs = clf.predict_proba(input_vector)
    confidence = max(probs[0])

    if confidence < 0.4:
        return "I'm not sure I understood. Can you rephrase?"

    tag = clf.predict(input_vector)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

counter = 0

# ---------- UI ----------
def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":

        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["User Input", "Chatbot Response", "Timestamp"]
                )

        counter += 1
        user_input = st.text_input("You:", key=f"user_{counter}")

        if user_input:
            response = chatbot(user_input)

            st.text_area(
                "Chatbot:",
                value=response,
                height=120,
                key=f"bot_{counter}"
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("chat_log.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [user_input, response, timestamp]
                )

    elif choice == "Conversation History":
        st.header("Conversation History")

        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    st.write(f"User: {row[0]}")
                    st.write(f"Bot: {row[1]}")
                    st.write(f"Time: {row[2]}")
                    st.markdown("---")

    else:
        st.write("NLP Chatbot using TF-IDF and Logistic Regression.")

if __name__ == "__main__":
    main()