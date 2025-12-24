import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)
st.markdown("""
<style>

/* Text area */
.stTextArea textarea {
    background-color: #020617;
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    color: #cbd5f5;
    font-size: 13px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df["review"] = df["review"].apply(clean_text)
    return df
data = load_data()
X = data["review"]
y = data["sentiment"]
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>🎬 Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Understand audience reaction using NLP</p>", unsafe_allow_html=True)
st.info(f"📊 **Model Accuracy:** {accuracy*100:.2f}%")
review_input = st.text_area(
    "✍️ Enter a movie review:",
    height=160,
    placeholder="The movie was absolutely brilliant with stunning visuals..."
)
if st.button("🎭 Analyze Audience Reaction"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        cleaned = clean_text(review_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = np.max(probs) * 100
        st.markdown("---")
        if pred == 1 and confidence > 75:
            st.success("🌟 **Audience Loved It!**")
            st.image("https://cdn-icons-png.flaticon.com/512/616/616430.png", width=120)
        elif pred == 1:
            st.info("🙂 **Mostly Positive Response**")
            st.image("https://cdn-icons-png.flaticon.com/512/742/742751.png", width=120)
        elif pred == 0 and confidence > 75:
            st.error("💔 **Audience Disliked It**")
            st.image("https://cdn-icons-png.flaticon.com/512/742/742752.png", width=120)
        else:
            st.warning("😕 **Mixed / Slightly Negative Reaction**")
            st.image("https://cdn-icons-png.flaticon.com/512/742/742774.png", width=120)
        st.markdown(f"### 🔐 Confidence Level: `{confidence:.2f}%`")
st.markdown(
    "<div class='footer'>Built using NLP, TF-IDF & Logistic Regression</div>",
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
