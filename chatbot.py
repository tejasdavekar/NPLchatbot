import os
import json
import ssl
import nltk
import datetime
import csv
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context 
nltk.data.path.append(os.path.abspath("nltk_data")) 
nltk.download('punkt')

file_path = os.path.abspath("intents.json")
with open(file_path, 'r') as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform (patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text) [0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.title("Chatbot using NLP")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the chat.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime(f"%d-%m-%Y %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with st.expander("Click to see Conversation History"):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input using Natural Language Processing (NLP) techniques.")

        st.subheader("**Project Overview:**")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labelled intents.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based interactive chatbot.
        """)

        st.subheader("**Dataset:**")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The dataset consists of:
        - **Intents**: The intent of the user input (e.g. "greeting", "budget", "about")
        - **Entities**: The entities extracted from user input (e.g. "Hi", "How do I create a budget?")
        - **Text**: The user input text.
        """)

        st.subheader("**Streamlit Chatbot Interface:**")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input for user messages and displays chatbot responses interactively.")

        st.subheader("**Conclusion:**")

        st.write("In this project, a chatbot is built that can understand and respond to user inputs using NLP techniques and Logistic Regression. The interactive interface built with Streamlit makes the chatbot user-friendly and accessible.")

if __name__ == '__main__':
    main()
