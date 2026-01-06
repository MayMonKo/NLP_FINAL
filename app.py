import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Load resources
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load model artifacts
# -----------------------------
with open("model/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model/logreg.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------
# UI
# -----------------------------
st.title("✈️ Airline Review Sentiment Analysis")
st.write("Binary sentiment classification using Logistic Regression + TF-IDF")

airline = st.selectbox(
    "Select Airline",
    ["Qatar Airways", "Turkish Airlines"]
)

review = st.text_area("Enter a customer review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        clean = preprocess_text(review)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ Positive Sentiment")
        else:
            st.error("❌ Negative Sentiment")
