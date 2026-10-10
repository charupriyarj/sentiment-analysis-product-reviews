import json
import os

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "sentiment_pipeline.joblib"
METRICS_PATH = "metrics.json"

st.set_page_config(
    page_title="ReviewSense AI",
    page_icon="📝",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7FC;
    }

    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #102A43;
        font-weight: 750;
    }

    h2, h3 {
        color: #163A63;
    }

    .hero {
        background: linear-gradient(135deg, #102A43, #1976D2);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(16, 42, 67, 0.12);
    }

    .hero h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #EAF4FF;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #DCE6F1;
        margin-bottom: 15px;
    }

    .result-card {
        background: white;
        border-left: 5px solid #1976D2;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 3px 12px rgba(16, 42, 67, 0.06);
    }

    .result-label {
        color: #526579;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }

    .result-value {
        color: #102A43;
        font-size: 1.5rem;
        font-weight: 700;
    }

    div.stButton > button {
        background-color: #1565C0;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 10px 18px;
        font-weight: 600;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background-color: #0D47A1;
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #DCE6F1;
        padding: 12px;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #DCE6F1;
        border-radius: 10px;
    }

    .footer {
        color: #627D98;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📝 ReviewSense AI</h1>
        <p>Sentiment Analysis of Product Reviews</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter a product review and let the trained machine learning "
    "model predict whether its sentiment is Positive, Negative, or Neutral."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Please check your GitHub repository."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------

st.divider()

st.subheader("🔍 Analyze a Product Review")

EXAMPLES = {
    "Positive example":
        "The product quality is excellent and delivery was fast.",

    "Negative example":
        "The product stopped working after two days.",

    "Neutral example":
        "The product is average and works as expected."
}

choice = st.selectbox(
    "Try an example (optional)",
    ["-- Write my own review --"] + list(EXAMPLES.keys())
)

default_text = EXAMPLES.get(choice, "")

review = st.text_area(
    "Enter a product review:",
    value=default_text,
    height=130,
    placeholder="Type your product review here..."
)

predict_clicked = st.button(
    "🔎 Predict Sentiment",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION AND RESULTS
# --------------------------------------------------

if predict_clicked:

    if not review.strip():
        st.warning("Please enter a product review first.")

    else:
        try:
            prediction = model.predict([review])[0]
            probabilities = model.predict_proba([review])[0]

            sentiment_icons = {
                "Positive": "😊",
                "Negative": "😞",
                "Neutral": "😐"
            }

            icon = sentiment_icons.get(str(prediction), "📊")

            st.divider()
            st.subheader("📌 Prediction Result")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Sentiment
                    </div>
                    <div class="result-value">
                        {icon} {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------
            # PROBABILITY CHART
            # ------------------------------------------

            st.subheader("📊 Sentiment Probability")

            probability_df = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability (%)": probabilities * 100
            })

            st.bar_chart(
                probability_df.set_index("Sentiment"),
                y="Probability (%)"
            )

            st.caption(
                "Probabilities represent the model's estimated confidence. "
                "They are not guarantees of correctness."
            )

            # ------------------------------------------
            # MODEL PERFORMANCE
            # ------------------------------------------

            st.divider()
            st.subheader("📈 Model Performance")

            if os.path.exists(METRICS_PATH):

                with open(METRICS_PATH, "r") as file:
                    metrics = json.load(file)

                with st.expander(
                    "Model performance (on held-out test data)",
                    expanded=False
                ):

                    st.write(
                        "These metrics summarize the model's performance "
                        "on the test dataset. They remain the same for "
                        "individual reviews."
                    )

                    # Accuracy and macro metrics

                    st.markdown("#### Overall and Macro Metrics")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Accuracy",
                        f"{metrics['accuracy']:.3f}"
                    )

                    col2.metric(
                        "Macro F1-Score",
                        f"{metrics['macro_f1']:.3f}"
                    )

                    col3, col4 = st.columns(2)

                    col3.metric(
                        "Macro Precision",
                        f"{metrics['macro_precision']:.3f}"
                    )

                    col4.metric(
                        "Macro Recall",
                        f"{metrics['macro_recall']:.3f}"
                    )

                    st.markdown("#### Weighted Metrics")

                    col5, col6 = st.columns(2)

                    col5.metric(
                        "Weighted Precision",
                        f"{metrics['weighted_precision']:.3f}"
                    )

                    col6.metric(
                        "Weighted Recall",
                        f"{metrics['weighted_recall']:.3f}"
                    )

                    st.metric(
                        "Weighted F1-Score",
                        f"{metrics['weighted_f1']:.3f}"
                    )

                    st.caption(
                        f"Training reviews: {metrics['n_train']:,} | "
                        f"Test reviews: {metrics['n_test']:,}"
                    )

            else:
                st.warning(
                    "The metrics.json file was not found. "
                    "Please check that it exists in your repository."
                )

        except Exception as error:
            st.error(
                f"An error occurred while analyzing the review: {error}"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        ReviewSense AI | Sentiment Analysis of Product Reviews<br>
        Model: TF-IDF + Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)