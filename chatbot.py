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

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    st.set_page_config(page_title="Chatbot using NLP", page_icon="🤖")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        nav = st.radio("", options=["**💬 Chat**", "**📂 Chat History**", "**💡 About**"], horizontal=True)

    st.markdown("<h1 style='text-align: center;'>🤖 Chatbot using Natural Language Processing</h1>", unsafe_allow_html=True)

    if nav == "**💬 Chat**":
        st.markdown("<h2 style='text-align: center;'>✨ Let's Chat!</h2>", unsafe_allow_html=True)
        st.write("Type your message below to interact with the chatbot.")
        
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("**Your Message:**", placeholder="Type something here...", help="Enter your message.")

        if user_input:
            response = chatbot(user_input)
            st.markdown(f"**🤖 Bot:** {response}")

            timestamp = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.success("Thank you for chatting! Have a great day!")
                st.stop()

    elif nav == "**📂 Chat History**":
        st.markdown("<h2 style='text-align: center;'>📂 Chat History</h2>", unsafe_allow_html=True)
        st.write("Review your past conversations below:")

        if os.path.exists('chat_log.csv'):
            with st.expander("View Chat History"):
                with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)
                    for row in csv_reader:
                        st.markdown(f"**🧑‍💻 You:** {row[0]}")
                        st.markdown(f"**🤖 Bot:** {row[1]}")
                        st.caption(f"🕒 Timestamp: {row[2]}")
                        st.markdown("---")
        else:
            st.warning("No chat history available.")

    elif nav == "**💡 About**":
        st.markdown("<h2 style='text-align: center;'>💡 About This Chatbot</h2>", unsafe_allow_html=True)
        st.write("""This chatbot uses advanced NLP techniques and Machine Learning to provide meaningful responses to user inputs. The interface is designed for ease of use and professional interaction.""")

        st.markdown("### 🚀 Features")
        st.write("""- **Dynamic Conversations**: Enjoy seamless interactions. - **Chat History**: Review past conversations for reference. - **Responsive UI**: A professional and user-friendly layout.""")

        st.markdown("### 🛠️ Technical Details")
        st.write("""1. **Text Vectorization**: User input is processed with `TfidfVectorizer`. 2. **Intent Classification**: Logistic Regression identifies user intent. 3. **Response Selection**: Predefined responses are chosen based on intent.""")

if __name__ == '__main__':
    main()
