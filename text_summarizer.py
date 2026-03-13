import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )
    return summarizer


def summarize_text(text, max_length=150, min_length=40):
    summarizer = load_model()

    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    return result[0]["summary_text"]
