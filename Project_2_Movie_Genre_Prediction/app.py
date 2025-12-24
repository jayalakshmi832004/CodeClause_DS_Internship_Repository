import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Movie Genre Prediction",
    page_icon="🎞️",
    layout="centered"
)

st.markdown("""
<style>
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}

.block-container {
    padding-top: 1rem;
}

.main-container {
    background: rgba(0,0,0,0.75);
    padding: 30px;
    border-radius: 16px;
}

.stTextArea textarea {
    background-color: #020617;
    color: white;
}
</style>
""", unsafe_allow_html=True)


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)


@st.cache_data
def load_data():
    df = pd.read_csv("wiki_movie_plots_deduped.csv")
    df = df[["Plot", "Genre"]].dropna()

    # Take only first genre
    df["Genre"] = df["Genre"].str.split(",").str[0].str.strip().str.lower()

    # Exact genres to predict
    valid_genres = [
        "action", "adventure", "comedy", "drama",
        "horror", "romance", "thriller",
        "western", "crime", "mystery"
    ]

    df = df[df["Genre"].isin(valid_genres)]

    # Balance dataset
    df = df.groupby("Genre").head(1200)

    df["Plot"] = df["Plot"].apply(clean_text)
    return df

data = load_data()


@st.cache_resource
def train_model(data):
    X = data["Plot"]
    y = data["Genre"]

    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1,2),
        min_df=5
    )

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model(data)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎞️ Movie Genre Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Predict movie genre from plot summary using NLP</p>",
    unsafe_allow_html=True
)

st.info(f"📊 **Model Accuracy:** {accuracy*100:.2f}%")

plot_input = st.text_area(
    "✍️ Enter movie plot summary:",
    height=180,
    placeholder="A police officer investigates a mysterious murder that leads to a dark conspiracy..."
)

if st.button("🎬 Predict Genre"):
    if plot_input.strip() == "":
        st.warning("⚠️ Please enter a plot summary.")
    else:
        cleaned_plot = clean_text(plot_input)
        vec = vectorizer.transform([cleaned_plot])
        prediction = model.predict(vec)[0]

        st.markdown("---")
        st.success(f"🎥 **Predicted Genre:** {prediction.capitalize()}")

st.markdown("</div>", unsafe_allow_html=True)
