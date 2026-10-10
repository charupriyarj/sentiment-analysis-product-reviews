import json
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "sentiment_pipeline.joblib"
METRICS_PATH = "metrics.json"

st.set_page_config(
    page_title="ReviewSense AI",
    page_icon="🛍️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #102A43, #1D4E89);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
}

.hero p {
    color: #E3EEFF;
    font-size: 17px;
}

div.stButton > button {
    background-color: #173F70;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #245A91;
    color: white;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #DCE6F1;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Header
st.markdown("""
<div class="hero">
    <h1>🛍️ ReviewSense AI</h1>
    <p>
        Understand customer opinions through
        Natural Language Processing and Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' not found. "
        "Please check your GitHub repository."
    )
    st.stop()

model = load_model()
# Load and display model performance metrics
st.divider()
st.header("📊 Model Performance")

METRICS_PATH = "metrics.json"

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Macro Precision", f"{metrics['macro_precision'] * 100:.2f}%")
    col3.metric("Macro Recall", f"{metrics['macro_recall'] * 100:.2f}%")
    col4.metric("Macro F1-Score", f"{metrics['macro_f1'] * 100:.2f}%")

    col5, col6, col7 = st.columns(3)

    col5.metric("Weighted Precision",
                f"{metrics['weighted_precision'] * 100:.2f}%")
    col6.metric("Weighted Recall",
                f"{metrics['weighted_recall'] * 100:.2f}%")
    col7.metric("Weighted F1-Score",
                f"{metrics['weighted_f1'] * 100:.2f}%")

    st.caption(
        f"Training samples: {metrics['n_train']:,} | "
        f"Testing samples: {metrics['n_test']:,}"
    )
else:
    st.warning("Performance metrics file (metrics.json) was not found.")

# Dashboard introduction
st.subheader("📊 Review Analysis Dashboard")
st.write(
    "Enter a product review below to explore its predicted sentiment."
)

col1, col2 = st.columns([3, 2])

EXAMPLES = {
    "Positive example": (
        "The product quality is excellent and delivery was fast."
    ),
    "Negative example": (
        "The product stopped working after two days."
    ),
    "Neutral example": (
        "The product is average and works as expected."
    ),
}

with col1:
    st.markdown("### ✍️ Analyze a Review")

    choice = st.selectbox(
        "Try a sample review",
        ["-- Write my own --"] + list(EXAMPLES.keys())
    )

    default_text = EXAMPLES.get(choice, "")

    review = st.text_area(
        "Product review",
        value=default_text,
        height=160,
        placeholder="Type or paste a product review here..."
    )

    predict_clicked = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

with col2:
    st.markdown("### 💡 What this app does")

    st.info(
        "This application uses a trained machine learning "
        "pipeline to classify product reviews."
    )

    st.markdown("""
    **Available sentiment classes**
    
    😊 Positive
    
    😐 Neutral
    
    😞 Negative
    """)

if predict_clicked:
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            pred = model.predict([review])[0]
            proba = model.predict_proba([review])[0]

            icons = {
                "Positive": "😊",
                "Negative": "😞",
                "Neutral": "😐"
            }

            st.markdown("---")
            st.subheader("📌 Analysis Result")

            st.metric(
                "Predicted Sentiment",
                f"{icons.get(str(pred), '📝')} {pred}"
            )

            probs = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": proba
            })

            probs["Probability (%)"] = (
                probs["Probability"] * 100
            ).round(2)

            st.markdown("#### Sentiment Probability")
            st.bar_chart(
                probs.set_index("Sentiment")["Probability"]
            )

            st.caption(
                "Probabilities represent the model's estimates, "
                "not guaranteed correctness."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption(
    "ReviewSense AI | Product Review Sentiment Analysis | "
    "Built with Python, Scikit-learn and Streamlit"
)