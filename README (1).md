# Sentiment Analysis of Product Reviews

Classifies a product review as **Positive**, **Negative** or **Neutral** using
TF-IDF features and Logistic Regression, served through a Streamlit web app.

## Dataset
**Amazon Fine Food Reviews** (Kaggle, CC0 / public domain; originally from
McAuley & Leskovec, Stanford SNAP). ~568,000 reviews with 1-5 star ratings.

Only two columns are used:
| Column | Meaning |
|--------|---------|
| `Text` | The full review written by the customer |
| `Score` | Star rating, 1 to 5 |

**Labeling approach:** the dataset has no sentiment column, so labels are derived
from the star rating: 1-2 = Negative, 3 = Neutral, 4-5 = Positive.
Limitation: a rating is a proxy for sentiment. Some 3-star reviews are mixed or
slightly positive, so Neutral is the hardest class. State this in your report.

## Project files
| File | Purpose |
|------|---------|
| `train.py` | Cleans data, trains the pipeline, evaluates it, saves the model |
| `app.py` | Streamlit web app that loads the saved model |
| `requirements.txt` | Libraries for Streamlit Cloud |
| `sentiment_pipeline.joblib` | Trained model (created by `train.py`) |
| `metrics.json` | Test-set metrics (created by `train.py`) |

## Step 1: Train in Google Colab
1. Download `Reviews.csv` from Kaggle ("Amazon Fine Food Reviews"; a free Kaggle
   account is needed). Unzip it.
2. Open https://colab.research.google.com, create a new notebook.
3. Click the folder icon on the left, then upload `Reviews.csv` and `train.py`.
4. Run in a cell:
```python
!python train.py --data Reviews.csv --sample 100000
```
(`--sample 0` uses all reviews; it is slower but fine on Colab.)
5. When it finishes, download `sentiment_pipeline.joblib`, `metrics.json`,
   `confusion_matrix.png` and `class_distribution.png` from the file panel.
   **Use the numbers printed by your own run in your report.**

## Step 2: Test locally (optional, VS Code)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Keep `sentiment_pipeline.joblib` and `metrics.json` in the same folder as `app.py`.

## Step 3: Put it on GitHub
Create a new public repository, then upload: `app.py`, `train.py`,
`requirements.txt`, `README.md`, `sentiment_pipeline.joblib`, `metrics.json`,
and the two PNG plots. Do **not** upload `Reviews.csv` (it is large and not
needed for deployment).

## Step 4: Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, choose your repository, branch `main`, main file `app.py`.
3. Click **Deploy**. After a few minutes you get a public link.

If you see a scikit-learn version error, pin the same version Colab used
(`import sklearn; print(sklearn.__version__)`) in `requirements.txt`, for example
`scikit-learn==1.x.x`. A model saved with one version should be loaded with the same one.

## Design decisions (for your viva)
- **Negation kept:** stop words are NOT removed and the tokenizer keeps "don't",
  so "not good" is not turned into "good". Bigrams (`ngram_range=(1, 2)`) let the
  model learn "not good" as a feature.
- **No data leakage:** TF-IDF is inside the Pipeline, so it is fitted only on the
  training split. Duplicate reviews are removed before splitting.
- **Class imbalance:** most Amazon reviews are 5-star, so `class_weight="balanced"`
  is used and macro-averaged metrics are reported alongside accuracy.
- **Reproducibility:** `random_state=42`, stratified 80/20 split.
