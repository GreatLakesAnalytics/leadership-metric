# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# Load trained NLP model
model = load("nlp_leadership_model.pkl")

st.set_page_config(page_title="Leadership Trait Evaluator", layout="centered")
st.title("🧠 NLP-Based Leadership Trait Evaluator")

st.markdown("""
This tool uses a trained machine learning model to score leadership traits from 3–4 short open-ended answers.  
Paste responses into the box below, and it will rate on a 1–10 scale:
- **Productivity**
- **Professionalism**
- **Communication**
- **Effectiveness**
- **Overall Leadership**
""")

response = st.text_area("✍️ Paste all responses here (combined into one string):", height=200)

if st.button("🔍 Evaluate"):
    if not response.strip():
        st.warning("Please enter a response before scoring.")
    else:
        prediction = model.predict([response])[0]
        traits = ["Productivity", "Professionalism", "Communication", "Effectiveness", "Overall Leadership"]
        
        df = pd.DataFrame({"Trait": traits, "Score": prediction})
        st.subheader("📊 Results")
        st.dataframe(df.style.format({"Score": "{:.2f}"}))

        # Plot chart
        fig, ax = plt.subplots()
        df.plot(kind="bar", x="Trait", y="Score", ax=ax, legend=False, ylim=(0, 10), color="skyblue")
        ax.set_ylabel("Score (1–10)")
        ax.set_title("Leadership Evaluation")
        st.pyplot(fig)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Scores (CSV)", csv, "leadership_scores.csv", "text/csv")
