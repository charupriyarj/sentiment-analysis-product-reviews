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