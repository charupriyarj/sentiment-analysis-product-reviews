"""
StudyMate - An AI Study Assistant
----------------------------------
This app lets a student paste in study material and then:
1. Summarize it
2. Explain a difficult concept from it in simple language
3. Generate 5 quiz questions with answers
4. Create a personalized study plan

Everything is built with Streamlit (for the web interface) and the
Anthropic Claude API (for the AI features).
"""

import os
import streamlit as st
from anthropic import Anthropic

# -----------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------
# st.set_page_config controls the browser tab title/icon and layout.
# This must be the very first Streamlit command in the file.
st.set_page_config(
    page_title="StudyMate - AI Study Assistant",
    page_icon="📚",
    layout="centered",
)

# -----------------------------------------------------------------
# 2. LOAD THE API KEY SAFELY (never hardcoded)
# -----------------------------------------------------------------
# We look in two places for the key, in this order:
#   a) An environment variable called ANTHROPIC_API_KEY
#   b) Streamlit's built-in "secrets" system (used on Streamlit Cloud)
# This means the key is NEVER typed directly into the code.
def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # st.secrets works like a dictionary but only exists if a
    # secrets.toml file (or Streamlit Cloud secrets) is configured.
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


api_key = get_api_key()

if not api_key:
    st.error(
        "No API key found. Please set the ANTHROPIC_API_KEY environment "
        "variable (or add it to Streamlit secrets) before using StudyMate. "
        "See the README for instructions."
    )
    st.stop()  # Stops the app here so nothing below tries to run without a key

# Create one Claude client we can reuse for every request.
client = Anthropic(api_key=api_key)

# The model used for every request. Kept in one place so it's easy to change.
MODEL_NAME = "claude-sonnet-4-6"


# -----------------------------------------------------------------
# 3. HELPER FUNCTION - talk to Claude
# -----------------------------------------------------------------
# Every feature in this app sends one prompt to Claude and shows the
# reply. Instead of repeating that code four times, we write it once
# here and reuse it.
def ask_claude(prompt: str, max_tokens: int = 1024) -> str:
    """Send a prompt to Claude and return its text reply."""
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content is a list of content blocks; for a simple
        # text reply there is normally just one block, so we join them
        # in case there's more than one.
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
    except Exception as e:
        return f"⚠️ Something went wrong while contacting Claude: {e}"


# -----------------------------------------------------------------
# 4. APP TITLE AND STUDY MATERIAL INPUT
# -----------------------------------------------------------------
st.title("📚 StudyMate")
st.write("Paste your study material below, then pick what you'd like help with.")

study_material = st.text_area(
    "Your study material",
    height=250,
    placeholder="Paste your notes, textbook paragraph, or article here...",
)

# We use "tabs" so each feature has its own clean space instead of
# cluttering one long page.
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Summarize", "💡 Explain a Concept", "❓ Quiz Me", "🗓️ Study Plan"]
)

# -----------------------------------------------------------------
# 5. FEATURE 1 - SUMMARIZE
# -----------------------------------------------------------------
with tab1:
    st.subheader("Summarize your material")
    st.write("Get a short, clear summary of the text you pasted above.")

    if st.button("Generate Summary"):
        if not study_material.strip():
            st.warning("Please paste some study material above first.")
        else:
            with st.spinner("Summarizing..."):
                prompt = (
                    "You are a helpful study assistant. Summarize the following "
                    "study material for a student in clear, simple language. "
                    "Use short bullet points and keep it concise.\n\n"
                    f"Study material:\n{study_material}"
                )
                result = ask_claude(prompt)
            st.markdown(result)

# -----------------------------------------------------------------
# 6. FEATURE 2 - EXPLAIN A DIFFICULT CONCEPT
# -----------------------------------------------------------------
with tab2:
    st.subheader("Explain a difficult concept")
    st.write("Tell StudyMate which concept from your material confuses you.")

    concept = st.text_input(
        "Which concept or term would you like explained?",
        placeholder="e.g. Newton's Third Law",
    )

    if st.button("Explain This Concept"):
        if not study_material.strip():
            st.warning("Please paste some study material above first.")
        elif not concept.strip():
            st.warning("Please type the concept you want explained.")
        else:
            with st.spinner("Explaining..."):
                prompt = (
                    "You are a friendly tutor. Using the study material below as "
                    f"context, explain the concept '{concept}' in very simple, "
                    "beginner-friendly language. Use an everyday analogy if it "
                    "helps, and keep it short.\n\n"
                    f"Study material:\n{study_material}"
                )
                result = ask_claude(prompt)
            st.markdown(result)

# -----------------------------------------------------------------
# 7. FEATURE 3 - GENERATE A QUIZ
# -----------------------------------------------------------------
with tab3:
    st.subheader("Quiz yourself")
    st.write("Generate 5 quiz questions (with answers) based on your material.")

    if st.button("Generate Quiz"):
        if not study_material.strip():
            st.warning("Please paste some study material above first.")
        else:
            with st.spinner("Writing quiz questions..."):
                prompt = (
                    "You are a study assistant. Based on the study material below, "
                    "write exactly 5 quiz questions to test understanding. "
                    "Mix question types if appropriate (multiple choice or short "
                    "answer). After each question, show the correct answer clearly "
                    "labeled as 'Answer:'. Number the questions 1 to 5.\n\n"
                    f"Study material:\n{study_material}"
                )
                result = ask_claude(prompt, max_tokens=1500)
            st.markdown(result)

# -----------------------------------------------------------------
# 8. FEATURE 4 - PERSONALIZED STUDY PLAN
# -----------------------------------------------------------------
with tab4:
    st.subheader("Create a study plan")
    st.write("Tell StudyMate a bit about your situation to get a personalized plan.")

    days_available = st.number_input(
        "How many days do you have to study?", min_value=1, max_value=90, value=7
    )
    hours_per_day = st.number_input(
        "How many hours can you study per day?", min_value=1, max_value=12, value=2
    )
    goal = st.text_input(
        "What's your goal?", placeholder="e.g. Pass my biology exam"
    )

    if st.button("Generate Study Plan"):
        if not study_material.strip():
            st.warning("Please paste some study material above first.")
        else:
            with st.spinner("Building your study plan..."):
                prompt = (
                    "You are a supportive study coach. Based on the study material "
                    f"below, create a day-by-day study plan for {days_available} "
                    f"days, with about {hours_per_day} hour(s) of studying per day. "
                    f"The student's goal is: {goal or 'general understanding'}. "
                    "Break the material into manageable chunks, suggest short "
                    "breaks, and include a quick review day near the end. Format "
                    "the plan clearly with a heading for each day.\n\n"
                    f"Study material:\n{study_material}"
                )
                result = ask_claude(prompt, max_tokens=1500)
            st.markdown(result)

# -----------------------------------------------------------------
# 9. FOOTER
# -----------------------------------------------------------------
st.divider()
st.caption("StudyMate is powered by the Anthropic Claude API.")
