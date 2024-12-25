import os
import json
import csv
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

FILE_PATH = "intents.json"
CHAT_LOG_FILE = 'chat_log.csv'
MAX_ITER = 10000
STATE_CHAT = "💬 Chat"
STATE_HISTORY = "📂 History"
STATE_ABOUT = "💡 About"

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def train_model(intents):
    vectorizer = TfidfVectorizer()
    clf = LogisticRegression(random_state=0, max_iter=MAX_ITER)

    tags = []
    patterns = []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)
    
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)
    return vectorizer, clf

def chatbot(input_text, vectorizer, clf, intents):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def initialize_sidebar():
    st.sidebar.markdown("### 🎨 Select Theme:")
    if st.sidebar.button("🔆 Light Theme"):
        st.markdown('<meta http-equiv="refresh" content="0; URL=\'?embed_options=light_theme\'">', unsafe_allow_html=True)
        st.stop()
    if st.sidebar.button("🌑 Dark Theme"):
        st.markdown('<meta http-equiv="refresh" content="0; URL=\'?embed_options=dark_theme\'">', unsafe_allow_html=True)
        st.stop()
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📞 Contact:")
    st.sidebar.write("🔗 Connect with me on [`LinkedIn`](https://www.linkedin.com/in/tejas-davekar/)! ", unsafe_allow_html=True)
    st.sidebar.write("🔗 Contact me on [`email`](mailto:tejasdavekar@outlook.in)! ", unsafe_allow_html=True)
    st.sidebar.write("🔗 View my [`Resume`](https://drive.google.com/file/d/1CCkjom-CQDOUs-moAuXwSMKL_IidckQ8/view)! ", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chatbot using NLP", page_icon=".\\t.ico")

    if "selected_button" not in st.session_state:
        st.session_state.selected_button = STATE_CHAT

    initialize_sidebar()

    col1, col3, col5 = st.columns([1, 1, 1])
    with col1:
        chat_button = st.button(STATE_CHAT)
    with col3:
        history_button = st.button(STATE_HISTORY)
    with col5:
        about_button = st.button(STATE_ABOUT)

    if chat_button:
        st.session_state.selected_button = STATE_CHAT
    elif history_button:
        st.session_state.selected_button = STATE_HISTORY
    elif about_button:
        st.session_state.selected_button = STATE_ABOUT

    st.markdown("<h1 style='text-align: center;'>🤖 Chatbot using Natural Language Processing</h1>", unsafe_allow_html=True)

    intents = load_data(FILE_PATH)
    vectorizer, clf = train_model(intents)

    if st.session_state.selected_button == STATE_CHAT:
        st.markdown("<h2 style='text-align: center;'>✨ Let's Chat!</h2>", unsafe_allow_html=True)
        st.write("Type your message below to interact with the chatbot.")

        if not os.path.exists(CHAT_LOG_FILE):
            with open(CHAT_LOG_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("**Your Message:**", placeholder="Type something here...", help="Type your message and tap `Enter`.")

        if user_input:
            response = chatbot(user_input, vectorizer, clf, intents)
            st.markdown(f"**🤖 Bot:** {response}")

            timestamp = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
            with open(CHAT_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.success("Thank you for chatting! Have a great day!")
                st.stop()

    elif st.session_state.selected_button == STATE_HISTORY:
        st.markdown("<h2 style='text-align: center;'>📂 Chat History</h2>", unsafe_allow_html=True)
        st.write("Review your past conversations below:")

        if os.path.exists(CHAT_LOG_FILE):
            with st.expander("View Chat History"):
                with open(CHAT_LOG_FILE, 'r', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)
                    for row in csv_reader:
                        st.markdown(f"**🧑‍💻 You:** {row[0]}")
                        st.markdown(f"**🤖 Bot:** {row[1]}")
                        st.caption(f"🕒 Timestamp: {row[2]}")
                        st.markdown("---")

    elif st.session_state.selected_button == STATE_ABOUT:
        st.markdown("<h2 style='text-align: center;'>💡 About This Chatbot</h2>", unsafe_allow_html=True)
        st.write("""This chatbot uses advanced NLP techniques and Machine Learning to provide meaningful responses to user inputs. The interface is designed for ease of use and professional interaction.""")

        st.markdown("### 🚀 Features")
        st.write("""- **Dynamic Conversations**: Enjoy seamless interactions.""")
        st.write("""- **Chat History**: Review past conversations for reference.""")
        st.write("""- **Responsive UI**: A professional and user-friendly layout.""")

        st.markdown("### 🛠 Technical Details")
        st.write("""1. **Text Vectorization**: User input is processed with `TfidfVectorizer`.""")
        st.write("""2. **Intent Classification**: Logistic Regression identifies user intent.""")
        st.write("""3. **Response Selection**: Predefined responses are chosen based on intent.""")

        st.markdown("### 📞 Contact")
        st.write( """🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/tejas-davekar/)!""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
