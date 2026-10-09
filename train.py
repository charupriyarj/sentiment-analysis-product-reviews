"""
Train a 3-class sentiment model (Negative / Neutral / Positive) on Amazon product reviews.

Dataset: "Amazon Fine Food Reviews" (Kaggle, CC0 licence; original data from
McAuley & Leskovec, Stanford SNAP). File: Reviews.csv
Columns used: Text (review text) and Score (1-5 star rating).

Labeling approach (defensible and commonly used):
    Score 1-2 -> Negative, Score 3 -> Neutral, Score 4-5 -> Positive

Usage:
    python train.py --data Reviews.csv --sample 100000
"""
import argparse
import json
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
LABELS = ["Negative", "Neutral", "Positive"]


def score_to_label(score):
    if score <= 2:
        return "Negative"
    if score == 3:
        return "Neutral"
    return "Positive"


def basic_clean(text):
    """Light cleaning ONLY. We keep negations (not, never, don't) and keep
    stop words, because words like 'not' flip the sentiment of a review."""
    text = re.sub(r"<[^>]+>", " ", str(text))   # remove HTML tags like <br />
    text = re.sub(r"http\S+", " ", text)         # remove URLs
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    return text


def main(args):
    # ---------- 1. Load and inspect ----------
    df = pd.read_csv(args.data)
    print("Shape:", df.shape)
    print(df[["Score", "Text"]].head())
    print("\nMissing values:\n", df[["Score", "Text"]].isna().sum())

    # ---------- 2. Clean the data ----------
    df = df[["Text", "Score"]].dropna()
    df = df[df["Score"].isin([1, 2, 3, 4, 5])]             # drop invalid labels
    df["Text"] = df["Text"].map(basic_clean)
    df = df[df["Text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset="Text")                  # duplicate reviews cause leakage
    print(f"\nRemoved {before - len(df)} duplicate reviews")

    df["label"] = df["Score"].map(score_to_label)
    print("\nClass distribution (full data):\n", df["label"].value_counts())

    if args.sample and len(df) > args.sample:               # keep Colab fast
        frac = args.sample / len(df)
        df, _ = train_test_split(df, train_size=frac, stratify=df["label"],
                                 random_state=RANDOM_STATE)
        print(f"\nUsing a stratified sample of {len(df)} reviews")

    # plot class distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="label", data=df, order=LABELS)
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.close()

    # ---------- 3. Train/test split (stratified, reproducible) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        df["Text"], df["label"], test_size=0.2,
        stratify=df["label"], random_state=RANDOM_STATE)

    # ---------- 4. Pipeline: TF-IDF + Logistic Regression ----------
    # The pipeline fits TF-IDF on training data only (no leakage).
    # ngram_range=(1,2) lets the model see "not good" as its own feature.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w[\w']*\b",   # keeps words like "don't"
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=100000,
            sublinear_tf=True)),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",             # handles class imbalance
            random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    # ---------- 5. Evaluate on unseen test data ----------
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro    P/R/F1: {p:.4f} / {r:.4f} / {f1:.4f}")
    print(f"Weighted P/R/F1: {pw:.4f} / {rw:.4f} / {f1w:.4f}")
    report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0)
    print("\nClassification report:\n", report)

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # ---------- 6. Save ----------
    joblib.dump(pipeline, "sentiment_pipeline.joblib", compress=3)
    with open("metrics.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "macro_precision": p, "macro_recall": r, "macro_f1": f1,
            "weighted_precision": pw, "weighted_recall": rw, "weighted_f1": f1w,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        }, f, indent=2)
    print("\nSaved sentiment_pipeline.joblib, metrics.json and plots.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Reviews.csv")
    ap.add_argument("--sample", type=int, default=100000,
                    help="max reviews to use (0 = use all)")
    main(ap.parse_args())
