# app.py  -  FlipItNews: Automated News Categorization Dashboard
# Run with:  streamlit run app.py
import json
import re

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from nltk.corpus import stopwords

st.set_page_config(page_title="FlipItNews Classifier", page_icon="📰", layout="centered")


# ---------- 1. Load the trained model + settings (only once) ----------
@st.cache_resource
def load_everything():
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    model = tf.keras.models.load_model("rnn_model.keras")
    with open("artifacts.json") as f:
        art = json.load(f)
    return model, art, stop_words


model, art, stop_words = load_everything()
vocab, classes, MAX_LEN = art["vocab"], art["classes"], art["max_len"]


# ---------- 2. Same preprocessing as the notebook (must match!) ----------
def clean_text(text):
    text = text.lower()                                   # lowercase
    text = re.sub(r"[^a-z\s]", " ", text)                 # remove punctuation / numbers
    tokens = text.split()                                 # tokenization
    return [w for w in tokens if w not in stop_words and len(w) > 1]  # remove stop words


def predict_proba(texts):
    seqs = [[vocab.get(w, 1) for w in clean_text(t)] for t in texts]   # words -> numbers
    padded = tf.keras.utils.pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    return model.predict(padded, verbose=0)


# ---------- 3. Sample articles so the demo works instantly ----------
SAMPLES = {
    "-- write my own --": "",
    "Sport": "England beat Australia by five wickets in the final Test, with the captain scoring a "
             "brilliant century as the home crowd cheered. The team now leads the series and the "
             "coach praised the bowlers for a disciplined performance on the last day.",
    "Business": "Shares in the airline rose 4% after it reported higher quarterly profits and said "
                "falling fuel costs would boost revenues. Analysts expect the company to raise its "
                "full-year forecast, while the central bank held interest rates steady.",
    "Politics": "The prime minister faced questions in parliament over the new immigration bill. "
                "Opposition MPs said the government had failed to consult, and a vote is expected "
                "next week ahead of the general election.",
    "Tech": "The software firm has released a new smartphone operating system with improved "
            "security and faster mobile broadband support. Users can download the update for free, "
            "and developers say the new browser is much quicker.",
    "Entertainment": "The film won three awards at the ceremony in Los Angeles, including best "
                     "actress. The singer also performed her new album's title track, and the "
                     "director thanked the cast backstage.",
}

# ---------- 4. The web page ----------
st.title("📰 FlipItNews - Automatic News Categoriser")
st.write("Paste a news article and the trained **RNN (LSTM)** model predicts its category.")

with st.sidebar:
    st.header("About")
    st.write("Model: RNN (LSTM) trained on the BBC News dataset.")
    st.write("Categories: " + ", ".join(c.title() for c in classes))
    st.caption(f"Articles are cut/padded to {MAX_LEN} words.")

tab_single, tab_batch = st.tabs(["Single article", "Batch (CSV)"])

with tab_single:
    choice = st.selectbox("Try a sample article", list(SAMPLES.keys()))
    text = st.text_area("News text", value=SAMPLES[choice], height=200,
                        placeholder="Paste a news article here...")

    if st.button("Classify", type="primary"):
        if len(text.split()) < 10:
            st.warning("Please enter at least 10 words.")
        else:
            probs = predict_proba([text])[0]
            best = int(np.argmax(probs))
            st.success(f"Predicted category: **{classes[best].title()}**")
            st.metric("Confidence", f"{probs[best] * 100:.1f}%")
            if probs[best] < 0.6:
                st.info("Low confidence - this article may need a human editor to check it.")
            chart = pd.DataFrame({"Probability": probs}, index=[c.title() for c in classes])
            st.bar_chart(chart)

with tab_batch:
    st.write("Upload a CSV with a column named **text** to label many articles at once.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        if "text" not in data.columns:
            st.error("The CSV must contain a 'text' column.")
        else:
            probs = predict_proba(data["text"].astype(str).tolist())
            data["predicted_category"] = [classes[i] for i in probs.argmax(axis=1)]
            data["confidence"] = probs.max(axis=1).round(3)
            st.dataframe(data)
            st.download_button("Download results", data.to_csv(index=False), "predictions.csv")
