import streamlit as st
from joblib import load
import pandas as pd

# Load the trained model
model = load("nlp_leadership_model.pkl")

st.title("Leadership Sentiment Analyzer")
st.markdown("Enter a short leadership evaluation or feedback statement:")

# User input
user_input = st.text_area("Feedback Text", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Predict
        prediction = model.predict([user_input])[0]
        traits = ["Productivity", "Professionalism", "Communication", "Effectiveness", "Overall Leadership"]
        results = dict(zip(traits, prediction))

        st.subheader("Leadership Trait Scores (1–10 scale):")
        for trait, score in results.items():
            st.markdown(f"**{trait}:** {score:.2f}")
