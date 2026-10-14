# ScamSentry AI

An explainable SMS spam-detection prototype built with Python, Streamlit and scikit-learn.

## Features
- Trains a TF-IDF + Logistic Regression text classifier on the UCI SMS Spam Collection.
- Evaluates using a stratified hold-out test set.
- Shows accuracy, spam precision, spam recall, F1-score and a confusion matrix.
- Accepts live messages and displays a prediction plus simple, rule-based warning signs.
- Does not require an API key or GPU.

## Run locally

Python 3.10+ is recommended.

### Windows
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the public UCI SMS Spam Collection dataset and trains the model. An internet connection is required for the first run.

## Methodology
- Dataset: UCI SMS Spam Collection (CC BY 4.0)
- Duplicate messages are removed before the split.
- Train/test split: 80/20, stratified, fixed random seed (42).
- Features: TF-IDF word and bigram features.
- Model: Logistic Regression.
- Metrics: accuracy, spam precision, spam recall, F1-score and confusion matrix.

## Dataset citation
Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

## Important limitations
This is an educational prototype, not a production fraud detector. The corpus contains historical English SMS messages. Model confidence is not a guarantee, and rule-based warning signs are not proof of fraud. Verify suspicious messages through official channels.
