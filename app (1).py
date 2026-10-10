import json
import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ReviewSense AI",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "sentiment_pipeline.joblib"
METRICS_PATH = "metrics.json"
CONFUSION_PATH = "confusion_matrix.png"
DISTRIBUTION_PATH = "class_distribution.png"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# LOAD MODEL METRICS
# =========================================================

def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

    return {}


metrics = load_metrics()


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================================================
# EXAMPLE REVIEWS
# =========================================================

example_reviews = {
    "Positive Review": (
        "This product is amazing! The quality is excellent "
        "and it works perfectly. I am very happy with it."
    ),
    "Negative Review": (
        "Very disappointed with this product. The quality "
        "is poor and it stopped working after two days."
    ),
    "Neutral Review": (
        "The product arrived today. It looks as described "
        "and has the features mentioned in the listing."
    )
}


# =========================================================
# MAIN HEADER
# =========================================================

st.title("📊 ReviewSense AI")

st.markdown(
    """
    ### Sentiment Analysis of Product Reviews

    Analyze product reviews using Machine Learning and
    understand whether the expressed sentiment is positive,
    negative, or neutral.
    """
)

st.divider()


# =========================================================
# CHECK MODEL AVAILABILITY
# =========================================================

if model is None:

    st.error(
        "The trained model file was not found. "
        "Please ensure `sentiment_pipeline.joblib` "
        "is present in the project folder."
    )

    st.stop()


# =========================================================
# APPLICATION TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Analyze Reviews",
        "🕘 Recent History",
        "📈 Model Performance",
        "ℹ️ About Project"
    ]
)


# =========================================================
# TAB 1: ANALYZE REVIEWS
# =========================================================

with tab1:

    st.subheader("🔍 Analyze a Product Review")

    st.write(
        "Enter a product review below to predict its sentiment."
    )

    selected_example = st.selectbox(
        "Choose an example review (optional)",
        ["Custom Review"] + list(example_reviews.keys())
    )

    if selected_example != "Custom Review":

        default_review = example_reviews[selected_example]

    else:

        default_review = ""

    review = st.text_area(
        "Enter your review",
        value=default_review,
        height=150,
        placeholder=(
            "Example: The product quality is excellent "
            "and I am very satisfied."
        ),
        key="review_input"
    )

    analyze_button = st.button(
        "Analyze Sentiment",
        type="primary",
        use_container_width=True
    )


    # -----------------------------------------------------
    # PREDICT SENTIMENT
    # -----------------------------------------------------

    if analyze_button:

        if not review.strip():

            st.warning(
                "Please enter a review before analyzing."
            )

        else:

            try:

                prediction = str(
                    model.predict([review])[0]
                )

                probabilities = None

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        [review]
                    )[0]

                    if hasattr(model, "classes_"):

                        classes = model.classes_

                    elif hasattr(
                        model,
                        "named_steps"
                    ):

                        classes = None

                        for step in model.named_steps.values():

                            if hasattr(step, "classes_"):

                                classes = step.classes_

                        if classes is None:
                            classes = []

                    else:

                        classes = []

                else:

                    classes = []


                # Save the latest result

                result = {
                    "Review": review.strip(),
                    "Predicted Sentiment": prediction
                }

                st.session_state.last_result = result


                # Add prediction to session history

                st.session_state.history.insert(
                    0,
                    result.copy()
                )

                # Keep only the 10 most recent predictions

                st.session_state.history = (
                    st.session_state.history[:10]
                )


                # Store probability information

                if (
                    probabilities is not None
                    and len(classes) == len(probabilities)
                ):

                    probability_df = pd.DataFrame(
                        {
                            "Sentiment": [
                                str(label)
                                for label in classes
                            ],
                            "Probability": [
                                float(value)
                                for value in probabilities
                            ]
                        }
                    )

                    st.session_state.last_result[
                        "probability_df"
                    ] = probability_df


            except Exception as error:

                st.error(
                    f"Unable to analyze the review: {error}"
                )


    # -----------------------------------------------------
    # DISPLAY LATEST PREDICTION
    # -----------------------------------------------------

    if st.session_state.last_result is not None:

        result = st.session_state.last_result

        st.divider()

        st.subheader("🧠 Sentiment Analysis Result")

        prediction = result["Predicted Sentiment"]

        sentiment_lower = prediction.lower()

        if "positive" in sentiment_lower:

            st.success(
                f"Predicted Sentiment: {prediction}"
            )

        elif "negative" in sentiment_lower:

            st.error(
                f"Predicted Sentiment: {prediction}"
            )

        elif "neutral" in sentiment_lower:

            st.info(
                f"Predicted Sentiment: {prediction}"
            )

        else:

            st.info(
                f"Predicted Sentiment: {prediction}"
            )


        # -------------------------------------------------
        # PROBABILITY CHART
        # -------------------------------------------------

        if "probability_df" in result:

            st.subheader("📊 Prediction Confidence")

            probability_df = result["probability_df"].copy()

            probability_df["Probability (%)"] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.bar_chart(
                probability_df.set_index("Sentiment")[
                    "Probability (%)"
                ]
            )

            st.dataframe(
                probability_df[
                    ["Sentiment", "Probability (%)"]
                ],
                use_container_width=True,
                hide_index=True
            )


    # =====================================================
    # DYNAMIC SENTIMENT DISTRIBUTION
    # =====================================================

    st.divider()

    st.subheader("📊 Sentiment Distribution")

    st.caption(
        "This chart updates as you analyze reviews. "
        "It shows the predicted sentiments of the "
        "reviews analyzed during the current session."
    )

    if st.session_state.history:

        # Convert prediction history into a DataFrame

        distribution_df = pd.DataFrame(
            st.session_state.history
        )

        # Count each predicted sentiment

        sentiment_counts = (
            distribution_df["Predicted Sentiment"]
            .value_counts()
        )

        # Include common sentiment categories, even if their
        # count is zero.

        standard_labels = [
            "Positive",
            "Negative",
            "Neutral"
        ]

        existing_labels = sentiment_counts.index.tolist()

        all_labels = standard_labels + [
            label
            for label in existing_labels
            if label not in standard_labels
        ]

        sentiment_counts = sentiment_counts.reindex(
            all_labels,
            fill_value=0
        )

        # Prepare chart data

        chart_df = (
            sentiment_counts
            .rename_axis("Sentiment")
            .reset_index(name="Number of Reviews")
        )

        # Display dynamic bar chart

        st.bar_chart(
            chart_df.set_index("Sentiment")
        )

        # Display exact counts

        st.subheader("Sentiment Counts")

        st.dataframe(
            chart_df,
            use_container_width=True,
            hide_index=True
        )

        total_reviews = len(distribution_df)

        st.metric(
            "Total Reviews Analyzed",
            total_reviews
        )

    else:

        st.info(
            "No reviews analyzed yet. Enter a review and "
            "click 'Analyze Sentiment' to generate the chart."
        )


# =========================================================
# TAB 2: RECENT HISTORY
# =========================================================

with tab2:

    st.subheader("🕘 Recent Review History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df[
                ["Review", "Predicted Sentiment"]
            ],
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.session_state.last_result = None

            st.rerun()

    else:

        st.info(
            "No reviews have been analyzed in this session yet."
        )


# =========================================================
# TAB 3: MODEL PERFORMANCE
# =========================================================

with tab3:

    st.subheader("📈 Model Performance")

    if metrics:

        # Display key metrics

        col1, col2, col3 = st.columns(3)

        accuracy = metrics.get("accuracy")

        if accuracy is not None:

            col1.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        macro_f1 = metrics.get("macro_f1")

        if macro_f1 is not None:

            col2.metric(
                "Macro F1-Score",
                f"{macro_f1:.4f}"
            )

        weighted_f1 = metrics.get("weighted_f1")

        if weighted_f1 is not None:

            col3.metric(
                "Weighted F1-Score",
                f"{weighted_f1:.4f}"
            )


        # Additional performance metrics

        st.subheader("Detailed Metrics")

        metrics_df = pd.DataFrame(
            [
                {
                    "Metric": key.replace("_", " ").title(),
                    "Value": value
                }
                for key, value in metrics.items()
            ]
        )

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )


        # Training and testing information

        n_train = metrics.get("n_train")

        n_test = metrics.get("n_test")

        if n_train is not None or n_test is not None:

            st.subheader("Dataset Information")

            col1, col2 = st.columns(2)

            if n_train is not None:

                col1.metric(
                    "Training Samples",
                    f"{n_train:,}"
                )

            if n_test is not None:

                col2.metric(
                    "Testing Samples",
                    f"{n_test:,}"
                )

    else:

        st.warning(
            "Model metrics are unavailable. "
            "Ensure that `metrics.json` is present "
            "in the project folder."
        )


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.divider()

    st.subheader("Confusion Matrix")

    if os.path.exists(CONFUSION_PATH):

        st.image(
            CONFUSION_PATH,
            caption="Confusion Matrix of the Trained Model",
            use_container_width=True
        )

    else:

        st.info(
            "The confusion matrix image was not found. "
            "Ensure `confusion_matrix.png` is present "
            "in the project folder."
        )


    # -----------------------------------------------------
    # ORIGINAL DATASET DISTRIBUTION
    # -----------------------------------------------------

    st.divider()

    st.subheader("Original Dataset Sentiment Distribution")

    if os.path.exists(DISTRIBUTION_PATH):

        st.image(
            DISTRIBUTION_PATH,
            caption=(
                "Sentiment distribution in the original dataset"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "The original dataset distribution image "
            "`class_distribution.png` was not found."
        )


# =========================================================
# TAB 4: ABOUT PROJECT
# =========================================================

with tab4:

    st.subheader("ℹ️ About ReviewSense AI")

    st.write(
        """
        **ReviewSense AI** is a machine learning application
        designed to analyze the sentiment expressed in
        product reviews.
        """
    )

    st.markdown(
        """
        ### Project Objectives

        - Analyze the text of product reviews.
        - Predict the sentiment expressed in a review.
        - Display prediction confidence when supported by
          the trained model.
        - Maintain a history of recently analyzed reviews.
        - Visualize the distribution of predicted sentiments.
        - Present model evaluation metrics and performance
          visualizations.

        ### Technologies Used

        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - Joblib
        - Machine Learning

        ### Project Files

        - `app (1).py` — Streamlit application
        - `sentiment_pipeline.joblib` — Trained model pipeline
        - `metrics.json` — Model evaluation metrics
        - `confusion_matrix.png` — Confusion matrix
        - `class_distribution.png` — Original dataset
          sentiment distribution
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ReviewSense AI | Sentiment Analysis of Product Reviews"
)