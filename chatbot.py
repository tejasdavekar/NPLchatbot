import os
import json
import ssl
import nltk
import random
import csv
import datetime
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

file_path = "intents.json"
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

    if "selected_button" not in st.session_state:
        st.session_state.selected_button = "💬 Chat"

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

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        chat_button = st.button("💬 Chat")
    with col3:
        history_button = st.button("📂 History")
    with col5:
        about_button = st.button("💡 About")

    if chat_button:
        st.session_state.selected_button = "💬 Chat"
    elif history_button:
        st.session_state.selected_button = "📂 History"
    elif about_button:
        st.session_state.selected_button = "💡 About"

    st.markdown("<h1 style='text-align: center;'>🤖 Chatbot using Natural Language Processing</h1>", unsafe_allow_html=True)

    if st.session_state.selected_button == "💬 Chat":
        st.markdown("<h2 style='text-align: center;'>✨ Let's Chat!</h2>", unsafe_allow_html=True)
        st.write("Type your message below to interact with the chatbot.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("**Your Message:**", placeholder="Type something here...", help="Type your message and tap `Enter`.")

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

    elif st.session_state.selected_button == "📂 History":
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

    elif st.session_state.selected_button == "💡 About":
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
