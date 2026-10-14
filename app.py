import io
import re
import zipfile
import urllib.request

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

st.set_page_config(
    page_title="ScamSentry AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#101b35,#243b63); color: white; margin-bottom: 1.2rem;}
    .hero h1 {color: white; margin-bottom: .3rem;}
    .hero p {color: #dce7ff; font-size: 1.05rem; margin-bottom: 0;}
    .small-note {color: #64748b; font-size: .88rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Downloading the public dataset and training the model for the first time…")
def train_model():
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "ScamSentry-AI-Student-Project/1.0"}
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist()
             if name.endswith("SMSSpamCollection")),
            None
        )
        if member is None:
            raise ValueError("The dataset file was not found inside the downloaded ZIP.")
        with archive.open(member) as raw_file:
            data = pd.read_csv(
                raw_file,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
                on_bad_lines="skip",
            )

    data = data.dropna(subset=["label", "message"]).copy()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"].str.len() > 0]
    data = data.drop_duplicates(subset=["message"]).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        "confusion": confusion_matrix(y_test, predictions, labels=["ham", "spam"]),
        "report": classification_report(y_test, predictions, labels=["ham", "spam"], target_names=["Legitimate (ham)", "Spam"], zero_division=0),
        "total": len(data),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "spam_count": int((data["label"] == "spam").sum()),
        "ham_count": int((data["label"] == "ham").sum()),
    }
    return model, data, metrics

def warning_signs(message):
    text = message.lower()
    checks = [
        ("Urgency or pressure", r"\b(urgent|immediately|act now|limited time|expires today|final warning)\b"),
        ("Prize or unexpected reward", r"\b(won|winner|prize|lottery|reward|claim your|free gift|cash prize)\b"),
        ("Link or web address", r"(https?://|www\.|bit\.ly|tinyurl\.com)"),
        ("Money or account request", r"\b(bank account|credit card|password|pin|otp|verify your account|payment|transfer money)\b"),
        ("Callback or contact pressure", r"\b(call now|reply now|click here|send us|confirm now)\b"),
    ]
    found = [label for label, pattern in checks if re.search(pattern, text)]
    return found

try:
    model, data, metrics = train_model()
except Exception as exc:
    st.error("The app could not download or read the public dataset. Check the internet connection and try again.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown("Dataset source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)")
    st.stop()

st.markdown("""
<div class="hero">
  <h1>🛡️ ScamSentry AI</h1>
  <p>Explainable SMS spam detection powered by Natural Language Processing and machine learning.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text features:** TF-IDF word and two-word patterns")
    st.write("**Training data:** UCI SMS Spam Collection")
    st.write("**Evaluation:** Stratified 80/20 train-test split")
    st.divider()
    st.caption("Prototype for educational use. Do not rely on it as the sole basis for financial or security decisions.")

tab_detect, tab_dashboard, tab_method = st.tabs(["🔎 Check a message", "📊 Model dashboard", "🧠 How it works"])

with tab_detect:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Analyze an SMS")
        st.write("Paste a message below. The model will estimate whether it resembles spam in its training data.")
        examples = {
            "Choose an example…": "",
            "Likely spam example": "URGENT! Congratulations, you won a free prize. Click https://example.com to claim your reward now!",
            "Likely legitimate example": "Hey, are we still meeting at the library at 4 pm? I will be there in ten minutes.",
            "Bank-themed example": "Your account needs verification. Click this link and enter your password immediately.",
        }
        chosen = st.selectbox("Try a sample", list(examples.keys()))
        default_text = examples[chosen]
        message = st.text_area(
            "Message text",
            value=default_text,
            height=155,
            placeholder="Paste the SMS message here…",
            max_chars=5000,
        )
        analyze = st.button("Analyze message", type="primary", use_container_width=True)

    with right:
        st.subheader("What you'll receive")
        st.markdown("- A model prediction and estimated probability")
        st.markdown("- Text-based warning signs to review")
        st.markdown("- A reminder to verify suspicious requests independently")
        st.info("A legitimate-looking message can still be a scam, and a spam-like message can be legitimate. Check the sender and never share OTPs or passwords.")

    if analyze:
        if not message.strip():
            st.warning("Enter a message first.")
        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)
            spam_probability = float(probabilities[classes.index("spam")])
            signs = warning_signs(message)

            st.divider()
            result_col, probability_col = st.columns(2)
            with result_col:
                if prediction == "spam":
                    st.error("### ⚠️ Model result: Spam-like")
                    st.write("The message resembles spam examples learned during training.")
                else:
                    st.success("### ✅ Model result: Likely legitimate")
                    st.write("The message resembles legitimate examples learned during training.")
            with probability_col:
                st.metric("Estimated spam probability", f"{spam_probability:.1%}")
                st.progress(int(round(spam_probability * 100)))
                st.caption("This is a model score, not a guarantee or a calibrated measure of real-world fraud risk.")

            st.subheader("Potential warning signs")
            if signs:
                for sign in signs:
                    st.markdown(f"- ⚠️ {sign}")
            else:
                st.write("No warning signs from the app's simple text rules were detected. This does not prove the message is safe.")

            if prediction == "spam" or signs:
                st.warning("Safety tip: don't click unexpected links or share passwords, PINs, or OTPs. Verify through the organization's official app or website.")
            else:
                st.info("Safety tip: check the sender and context before trusting any message.")

with tab_dashboard:
    st.subheader("Evaluation on held-out test data")
    st.write("The figures below are calculated on test examples excluded from model training.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Spam precision", f"{metrics['precision']:.1%}")
    c3.metric("Spam recall", f"{metrics['recall']:.1%}")
    c4.metric("Spam F1-score", f"{metrics['f1']:.1%}")

    st.caption("Precision: among messages predicted as spam, how many were spam in the test set. Recall: among actual spam messages, how many were detected. F1 balances precision and recall.")
    cm = metrics["confusion"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual legitimate", "Actual spam"],
        columns=["Predicted legitimate", "Predicted spam"],
    )
    st.subheader("Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)
    st.caption("Rows are actual labels; columns are model predictions.")
    st.subheader("Dataset overview")
    d1, d2, d3 = st.columns(3)
    d1.metric("Unique messages", f"{metrics['total']:,}")
    d2.metric("Training examples", f"{metrics['train_size']:,}")
    d3.metric("Test examples", f"{metrics['test_size']:,}")
    class_df = pd.DataFrame({
        "Class": ["Legitimate (ham)", "Spam"],
        "Messages": [metrics["ham_count"], metrics["spam_count"]],
    }).set_index("Class")
    st.bar_chart(class_df)
    with st.expander("View full classification report"):
        st.code(metrics["report"])

with tab_method:
    st.subheader("The machine-learning pipeline")
    st.markdown("""
    **1. Dataset:** Use labelled examples of legitimate and spam SMS messages from the UCI repository.

    **2. Preprocessing:** Remove invalid rows and duplicate messages, then split the data into training and test sets.

    **3. TF-IDF:** Convert text into numerical features. Words and two-word combinations that help distinguish messages receive different weights.

    **4. Logistic Regression:** Train a supervised classifier to estimate whether a message belongs to the spam or legitimate class.

    **5. Evaluation:** Test on data withheld from training and report accuracy, spam precision, spam recall, F1-score and a confusion matrix.

    **6. User-facing explanation:** Display simple, rule-based warning signs separately from the model prediction. These rules are not the model's explanation and can produce false alarms.
    """)
    st.subheader("Limitations and next steps")
    st.markdown("""
    - The training corpus is historical and English-language; modern scams and regional languages may behave differently.
    - A high test score does not guarantee performance on messages from new sources.
    - Warning-sign rules are simple indicators, not proof of fraud.
    - Future work could add multilingual datasets, URL reputation checks from approved sources, drift monitoring and user feedback.
    """)
    st.subheader("Dataset citation")
    st.markdown("Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). DOI: 10.24432/C5CC84.")

st.divider()
st.markdown('<p class="small-note">ScamSentry AI • Educational prototype • Always verify suspicious messages independently.</p>', unsafe_allow_html=True)
