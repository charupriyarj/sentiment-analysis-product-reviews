import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "sentiment_pipeline.joblib"
METRICS_PATH = "metrics.json"

st.set_page_config(page_title="Product Review Sentiment Analyzer", page_icon="📝")


@st.cache_resource
def load_model():
    """Load the saved pipeline once (not retrained on every start)."""
    return joblib.load(MODEL_PATH)


st.title("📝 Sentiment Analysis of Product Reviews")
st.write("Type a product review and the trained model will predict whether it is "
         "**Positive**, **Negative** or **Neutral**.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Run train.py first and "
             "place the generated file next to app.py.")
    st.stop()

model = load_model()

EXAMPLES = {
    "Positive example": "The product quality is excellent and delivery was fast.",
    "Negative example": "The product stopped working after two days.",
    "Neutral example": "The product is average and works as expected.",
}
choice = st.selectbox("Try an example (optional)", ["-- write my own --"] + list(EXAMPLES))
default_text = EXAMPLES.get(choice, "")
review = st.text_area("Enter a product review:", value=default_text, height=150)

if st.button("Predict Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        pred = model.predict([review])[0]
        proba = model.predict_proba([review])[0]
        icon = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(pred, "")
        st.subheader(f"Predicted Sentiment: {pred} {icon}")
        probs = pd.DataFrame({"Sentiment": model.classes_, "Probability": proba})
        st.bar_chart(probs.set_index("Sentiment"))
        st.caption("Probabilities show the model's confidence; they are not guarantees.")

with st.expander("Model performance (on held-out test data)"):
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        st.write(f"Accuracy: **{m['accuracy']:.3f}**  |  "
                 f"Macro F1: **{m['macro_f1']:.3f}**  |  "
                 f"Test reviews: {m['n_test']}")
    else:
        st.write("metrics.json not found.")

st.caption("Model: TF-IDF + Logistic Regression. Labels derived from star ratings "
           "(1-2 Negative, 3 Neutral, 4-5 Positive).")