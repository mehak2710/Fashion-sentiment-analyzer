import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Download VADER once
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample fashion reviews
sample_reviews = [
    "Absolutely love this dress! The fit is perfect and the fabric feels luxurious.",
    "Shoes were okay, not as comfortable as expected.",
    "Terrible stitching on the blouse. Wouldn’t recommend.",
    "This handbag is gorgeous and goes with everything I wear!",
    "The jeans are fine, but they faded after one wash.",
    "Amazing quality jacket! Worth every penny.",
    "Not happy with the color of the skirt. Looks different from the picture."
]

# Analyze sentiments
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# App title
st.title("🧵 Fashion Sentiment Analyzer")
st.markdown("Analyze sentiment in fashion product reviews using **VADER**.")

# Display sample reviews and sentiment
st.subheader("📦 Sample Reviews")
results = pd.DataFrame({
    "Review": sample_reviews,
    "Sentiment": [analyze_sentiment(r) for r in sample_reviews]
})
st.dataframe(results, use_container_width=True)

# User input
st.subheader("💬 Try Your Own Review")
user_input = st.text_area("Write a fashion review:")
if user_input:
    sentiment = analyze_sentiment(user_input)
    st.success(f"**Sentiment:** {sentiment}")
