import json
import random
import csv
import datetime
import os
import joblib
import nltk
import ssl
import streamlit as st
from streamlit_chat import message
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context 
nltk.data.path.append(os.path.abspath("nltk_data")) 
nltk.download('punkt')

file_path = "intents.json"
with open(file_path, 'r') as file:
    intents = json.load(file)

model_path = "chatbot_model.joblib"
vectorizer_path = "tfidf_vectorizer.joblib"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
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

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def main():
    st.set_page_config(page_title="Chatbot using NLP", page_icon="🤖", layout="centered")

    st.sidebar.markdown("### ☰ Menu:")
    chat_button = st.sidebar.button("💬 Chat")
    history_button = st.sidebar.button("📂 History")
    about_button = st.sidebar.button("💡 About")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🎨 Select Theme:")
    if st.sidebar.button("🔆 Light Mode"):
        st.markdown('<meta http-equiv="refresh" content="0; URL=\'?embed_options=light_theme\'">', unsafe_allow_html=True)
        st.stop()
    if st.sidebar.button("🌑 Dark Mode"):
        st.markdown('<meta http-equiv="refresh" content="0; URL=\'?embed_options=dark_theme\'">', unsafe_allow_html=True)
        st.stop()
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📞 Contact:")
    st.sidebar.write("🔗 Connect with me on [`LinkedIn`](https://www.linkedin.com/in/tejas-davekar/)! ", unsafe_allow_html=True)
    st.sidebar.write("🔗 Contact me on [`email`](mailto:tejasdavekar@outlook.in)! ", unsafe_allow_html=True)
    st.sidebar.write("🔗 View my [`Resume`](https://drive.google.com/file/d/1CCkjom-CQDOUs-moAuXwSMKL_IidckQ8/view)! ", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    if "selected_button" not in st.session_state:
        st.session_state.selected_button = "💬 Chat"
    if chat_button:
        st.session_state.selected_button = "💬 Chat"
    elif history_button:
        st.session_state.selected_button = "📂 History"
    elif about_button:
        st.session_state.selected_button = "💡 About"


    st.markdown("<h1 style='text-align: center;'>🤖 Chatbot using NLP</h1>", unsafe_allow_html=True)

    if st.session_state.selected_button == "💬 Chat":
        st.markdown("<h2 style='text-align: center;'>✨ Let's Chat!</h2>", unsafe_allow_html=True)

        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        if "messages" not in st.session_state:
            st.session_state.messages = []
        user_input = st.text_input("Your Message:", placeholder="Type something here.....", help="Type your message and tap `Enter`.")
        
        if user_input:
            bot_response = chatbot(user_input)
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            st.session_state.messages.append({"user": user_input, "bot": bot_response, "time": timestamp})

            with open("chat_log.csv", "a", newline="", encoding="utf-8") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([user_input, bot_response, timestamp])

        for msg in st.session_state.messages:
            message(msg["user"], is_user=True, key=f"user_{msg['time']}")
            message(msg["bot"], is_user=False, key=f"bot_{msg['time']}")

    elif st.session_state.selected_button == "📂 History":
        st.markdown("<h2 style='text-align: center;'>📂 Chat History</h2>", unsafe_allow_html=True)
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                next(csv_reader) 
                history = list(csv_reader)
            if history:
                with st.expander("View Chat History"):
                    for row in history:
                        st.markdown(f"**🧑‍💻 You:** {row[0]}")
                        st.markdown(f"**🤖 Bot:** {row[1]}")
                        st.caption(f"🕒 Timestamp: {row[2]}")
                        st.markdown("---")
            else:
                st.info("No chat history available.")
        else:
            st.info("No chat history found. Start chatting to create one!")

    elif st.session_state.selected_button == "💡 About":
        st.markdown("<h2 style='text-align: center;'>💡 About This Chatbot</h2>", unsafe_allow_html=True)
        st.write("""This chatbot uses advanced NLP techniques and Machine Learning to provide meaningful responses to user inputs. The interface is designed for ease of use and professional interaction.""")
        st.markdown("---")

        st.markdown("### 🚀 Features")
        st.write("""- **Dynamic Conversations**: Enjoy seamless interactions.""")
        st.write("""- **Chat History**: Review past conversations for reference.""")
        st.write("""- **Responsive UI**: A professional and user-friendly layout.""")
        st.markdown("---")

        st.markdown("### 🛠 Technical Details")
        st.write("""1. **Text Vectorization**: User input is processed with `TfidfVectorizer`.""")
        st.write("""2. **Intent Classification**: Logistic Regression identifies user intent.""")
        st.write("""3. **Response Selection**: Predefined responses are chosen based on intent.""")
        st.markdown("---")

        st.markdown("### 📞 Contact")
        st.write( """🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/tejas-davekar/)!""", unsafe_allow_html=True)
        st.markdown("---")

if __name__ == "__main__":
    main()
