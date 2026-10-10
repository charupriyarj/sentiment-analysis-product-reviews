import json
import os

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "sentiment_pipeline.joblib"
METRICS_PATH = "metrics.json"
CONFUSION_PATH = "confusion_matrix.png"

st.set_page_config(
    page_title="ReviewSense AI",
    page_icon="🛍️",
    layout="wide"
)


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7FC;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #102A43, #1976D2);
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #EAF4FF;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .result-card {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #DCE6F1;
        border-left: 6px solid #1976D2;
        margin: 10px 0 18px 0;
        box-shadow: 0 4px 12px rgba(16, 42, 67, 0.06);
    }

    .positive {
        border-left-color: #16A34A;
    }

    .negative {
        border-left-color: #DC2626;
    }

    .neutral {
        border-left-color: #EAB308;
    }

    .result-label {
        color: #526579;
        font-size: 0.9rem;
    }

    .result-value {
        color: #102A43;
        font-size: 1.7rem;
        font-weight: 750;
        margin-top: 5px;
    }

    h1, h2, h3 {
        color: #163A63;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #DCE6F1;
        padding: 14px;
        border-radius: 12px;
    }

    div.stButton > button {
        background-color: #1565C0;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
        padding: 10px 18px;
    }

    div.stButton > button:hover {
        background-color: #0D47A1;
        color: white;
    }

    div[data-testid="stExpander"] {
        background: white;
        border: 1px solid #DCE6F1;
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #627D98;
        font-size: 0.85rem;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL AND METRICS
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


if not os.path.exists(MODEL_PATH):
    st.error(
        "The trained model file was not found. "
        "Check that sentiment_pipeline.joblib is in your repository."
    )
    st.stop()


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


try:
    metrics = load_metrics()
except Exception as error:
    metrics = None
    st.warning(f"Could not read model metrics: {error}")


# =====================================================
# WELCOME SECTION
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛍️ ReviewSense AI</h1>
        <p><b>Understand customer opinions with Sentiment Analysis</b></p>
        <p>
            Analyze product reviews using Natural Language Processing
            and Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# NAVIGATION
# =====================================================

tab_home, tab_history, tab_performance, tab_about = st.tabs(
    [
        "🏠 Analyze Reviews",
        "🕘 Recent History",
        "📊 Model Performance",
        "ℹ️ About Project"
    ]
)


# =====================================================
# TAB 1: REVIEW ANALYSIS
# =====================================================

with tab_home:

    st.subheader("🔍 Analyze a Product Review")

    st.write(
        "Enter a review or select an example to see the model's prediction."
    )

    examples = {
        "Positive example":
            "The product quality is excellent and delivery was fast.",

        "Negative example":
            "The product stopped working after two days.",

        "Neutral example":
            "The product is average and works as expected."
    }

    choice = st.selectbox(
        "Try an example (optional)",
        ["-- Write my own review --"] + list(examples.keys())
    )

    review = st.text_area(
        "Enter your product review:",
        value=examples.get(choice, ""),
        height=130,
        placeholder="Type your product review here..."
    )

    predict_clicked = st.button(
        "🔎 Predict Sentiment",
        type="primary",
        use_container_width=True
    )

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------

    if predict_clicked:

        if not review.strip():
            st.warning("Please enter a review first.")

        else:
            try:
                prediction = str(model.predict([review])[0])

                confidence = None
                probability_records = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([review])[0]
                    classes = list(model.classes_)

                    confidence = float(
                        probabilities[classes.index(prediction)]
                    )

                    probability_records = [
                        {
                            "Sentiment": str(label),
                            "Probability (%)": round(
                                float(probability) * 100, 2
                            )
                        }
                        for label, probability
                        in zip(classes, probabilities)
                    ]

                # Save the latest result.

                st.session_state.last_result = {
                    "review": review.strip(),
                    "prediction": prediction,
                    "confidence": confidence,
                    "probability_records": probability_records
                }

                # Add the result to history.

                st.session_state.history.insert(
                    0,
                    {
                        "Review": review.strip(),
                        "Predicted Sentiment": prediction
                    }
                )

                # Keep only the latest 10 reviews.

                st.session_state.history = (
                    st.session_state.history[:10]
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    # -------------------------------------------------
    # DISPLAY RESULT
    # -------------------------------------------------

    result = st.session_state.last_result

    if result is not None:

        prediction = result["prediction"]
        confidence = result["confidence"]

        st.divider()
        st.subheader("🎯 Prediction Result")

        style_class = {
            "Positive": "positive",
            "Negative": "negative",
            "Neutral": "neutral"
        }.get(prediction, "")

        sentiment_display = {
            "Positive": "🟢 Positive Review",
            "Negative": "🔴 Negative Review",
            "Neutral": "🟡 Neutral Review"
        }.get(prediction, f"📊 {prediction}")

        explanation = {
            "Positive":
                "The model classified this review as expressing a positive opinion.",

            "Negative":
                "The model classified this review as expressing a negative opinion.",

            "Neutral":
                "The model classified this review as expressing a neutral or mixed opinion."
        }.get(
            prediction,
            "The model returned a sentiment classification."
        )

        st.markdown(
            f"""
            <div class="result-card {style_class}">
                <div class="result-label">Predicted Sentiment</div>
                <div class="result-value">{sentiment_display}</div>
                <p>{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------

        if confidence is not None:

            col1, col2 = st.columns(2)

            col1.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            col2.metric(
                "Predicted Class",
                prediction
            )

            st.caption(
                "Confidence is the model's estimated probability "
                "for the predicted class. It is not the same as "
                "accuracy and is not necessarily calibrated."
            )

        # ---------------------------------------------
        # SENTIMENT PROBABILITY CHART
        # ---------------------------------------------

        probability_records = result.get("probability_records")

        if probability_records is not None:

            probability_df = pd.DataFrame(probability_records)

            st.subheader("📊 Sentiment Probability")

            st.bar_chart(
                probability_df.set_index("Sentiment")
            )


# =====================================================
# TAB 2: RECENT HISTORY
# =====================================================

with tab_history:

    st.subheader("🕘 Recent Analysis History")

    st.write(
        "This history is stored only for the current app session. "
        "It is not permanently saved."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []
            st.session_state.last_result = None

            st.rerun()

    else:

        st.info(
            "No reviews analyzed in this session yet. "
            "Submit a review in the Analyze Reviews tab to begin."
        )


# =====================================================
# TAB 3: MODEL PERFORMANCE
# =====================================================

with tab_performance:

    st.subheader("📈 Model Performance")

    st.write(
        "These metrics summarize the trained model's performance "
        "on the held-out test dataset."
    )

    if metrics is not None:

        # First row of metrics.

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        c2.metric(