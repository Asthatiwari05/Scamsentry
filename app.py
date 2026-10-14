import io
import re
import time
import urllib.request
import zipfile

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION AND STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ScamSentry AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1200px;}
        .hero {padding: 1.5rem 1.7rem; border-radius: 18px;
               background: linear-gradient(120deg, #101b35, #243b63);
               color: white; margin-bottom: 1.2rem;}
        .hero h1 {color: white; margin-bottom: .3rem;}
        .hero p {color: #dce7ff; font-size: 1.05rem; margin-bottom: 0;}
        .small-note {color: #64748b; font-size: .88rem;}
        div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
                                     padding: 12px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# DATA LOADING, TRAINING, CROSS-VALIDATION AND EVALUATION
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the UCI dataset, tuning Naive Bayes with stratified cross-validation, and training models…")
def train_models():
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "ScamSentry-AI-Student-Project/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith("SMSSpamCollection")),
            None,
        )
        if member is None:
            raise ValueError("SMSSpamCollection was not found inside the downloaded ZIP.")
        with archive.open(member) as raw_file:
            data = pd.read_csv(
                raw_file,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
                on_bad_lines="skip",
            )

    # Clean invalid rows and duplicates.
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

    def make_vectorizer():
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        )

    # Spam is the positive class for model-selection metrics.
    spam_f1_scorer = make_scorer(
        f1_score, pos_label="spam", zero_division=0
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Tune Multinomial Naive Bayes alpha using only training data.
    # The vectorizer is inside the pipeline, preventing CV feature leakage.
    nb_pipeline = Pipeline(
        [
            ("tfidf", make_vectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )
    parameter_grid = {"classifier__alpha": [0.1, 0.5, 1.0, 2.0]}
    grid_search = GridSearchCV(
        estimator=nb_pipeline,
        param_grid=parameter_grid,
        scoring=spam_f1_scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    tuning_start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    tuning_time = time.perf_counter() - tuning_start
    naive_bayes_model = grid_search.best_estimator_
    nb_predictions = naive_bayes_model.predict(X_test)

    # Comparison model: Logistic Regression with the same text features and split.
    logistic_model = Pipeline(
        [
            ("tfidf", make_vectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logistic_start = time.perf_counter()
    logistic_model.fit(X_train, y_train)
    logistic_training_time = time.perf_counter() - logistic_start
    logistic_predictions = logistic_model.predict(X_test)

    def evaluate(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, pos_label="spam", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, pos_label="spam", zero_division=0
            ),
            "f1": f1_score(y_true, y_pred, pos_label="spam", zero_division=0),
            "confusion": confusion_matrix(y_true, y_pred, labels=["ham", "spam"]),
            "report": classification_report(
                y_true,
                y_pred,
                labels=["ham", "spam"],
                target_names=["Legitimate (ham)", "Spam"],
                zero_division=0,
            ),
        }

    nb_metrics = evaluate(y_test, nb_predictions)
    logistic_metrics = evaluate(y_test, logistic_predictions)

    cv_results = []
    for mean_score, std_score, params in zip(
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"],
        grid_search.cv_results_["params"],
    ):
        cv_results.append(
            {
                "Alpha": params["classifier__alpha"],
                "Mean CV spam F1": float(mean_score),
                "CV F1 standard deviation": float(std_score),
            }
        )

    metrics = {
        "total": len(data),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "spam_count": int((data["label"] == "spam").sum()),
        "ham_count": int((data["label"] == "ham").sum()),
        "naive_bayes": nb_metrics,
        "logistic": logistic_metrics,
        "best_alpha": float(grid_search.best_params_["classifier__alpha"]),
        "best_cv_f1": float(grid_search.best_score_),
        "cv_results": pd.DataFrame(cv_results),
        "tuning_time": tuning_time,
        "logistic_training_time": logistic_training_time,
        "comparison": {
            "Multinomial Naive Bayes (tuned)": {
                **{key: nb_metrics[key] for key in ["accuracy", "precision", "recall", "f1"]},
                "training_time": tuning_time,
            },
            "Logistic Regression": {
                **{key: logistic_metrics[key] for key in ["accuracy", "precision", "recall", "f1"]},
                "training_time": logistic_training_time,
            },
        },
    }

    # Primary prediction model is tuned Multinomial Naive Bayes.
    return naive_bayes_model, logistic_model, data, metrics


# ---------------------------------------------------------------------------
# RULE-BASED WARNING SIGNS (separate from ML prediction)
# ---------------------------------------------------------------------------
def warning_signs(message):
    text = message.lower()
    checks = [
        ("Urgency or pressure", r"\b(urgent|immediately|act now|limited time|expires today|final warning)\b"),
        ("Prize or unexpected reward", r"\b(won|winner|prize|lottery|reward|claim your|free gift|cash prize)\b"),
        ("Link or web address", r"(https?://|www\.|bit\.ly|tinyurl\.com)"),
        ("Money or account request", r"\b(bank account|credit card|password|pin|otp|verify your account|payment|transfer money)\b"),
        ("Callback or contact pressure", r"\b(call now|reply now|click here|send us|confirm now)\b"),
    ]
    return [label for label, pattern in checks if re.search(pattern, text)]


# ---------------------------------------------------------------------------
# NAIVE BAYES-SPECIFIC CONTRIBUTIONS
# For TF-IDF feature values x_j, this estimates each feature's contribution
# to the NB spam-vs-ham log score using log P(feature|spam)-log P(feature|ham).
# This is an explanatory approximation, not proof of fraud or causation.
# ---------------------------------------------------------------------------
def get_nb_prediction_terms(model, message, top_n=5):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    vector = vectorizer.transform([message])
    feature_names = vectorizer.get_feature_names_out()

    classes = list(classifier.classes_)
    spam_index = classes.index("spam")
    ham_index = classes.index("ham")

    log_prob_spam = classifier.feature_log_prob_[spam_index]
    log_prob_ham = classifier.feature_log_prob_[ham_index]
    log_likelihood_differences = log_prob_spam - log_prob_ham

    row = vector.tocsr()
    contributions = []
    for index, value in zip(row.indices, row.data):
        contribution = float(value * log_likelihood_differences[index])
        contributions.append((feature_names[index], contribution))

    toward_spam = sorted(
        [item for item in contributions if item[1] > 0],
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    toward_ham = sorted(
        [item for item in contributions if item[1] < 0],
        key=lambda item: item[1],
    )[:top_n]
    return toward_spam, toward_ham


# ---------------------------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------------------------
try:
    model, logistic_model, data, metrics = train_models()
except Exception as exc:
    st.error(
        "The app could not download the dataset or train the models. "
        "Check your internet connection and try again."
    )
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown(
        "Dataset source: [UCI SMS Spam Collection]"
        "(https://archive.ics.uci.edu/dataset/228/sms+spam+collection)"
    )
    st.stop()


# ---------------------------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ ScamSentry AI</h1>
        <p>Explainable SMS spam detection powered by Natural Language Processing and machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About the model")
    st.write("**Primary algorithm:** Tuned Multinomial Naive Bayes")
    st.write("**Comparison algorithm:** Logistic Regression")
    st.write("**Text features:** TF-IDF word unigrams and bigrams")
    st.write("**Tuning:** 5-fold StratifiedKFold, optimizing spam F1-score")
    st.write("**Training data:** UCI SMS Spam Collection")
    st.write("**Final evaluation:** Stratified 80/20 train-test split")
    st.divider()
    st.caption(
        "Educational prototype. Do not rely on it as the sole basis "
        "for financial or security decisions."
    )

tab_detect, tab_dashboard, tab_method = st.tabs(
    ["🔎 Check a message", "📊 Model dashboard", "🧠 How it works"]
)


# ---------------------------------------------------------------------------
# TAB 1: MESSAGE ANALYSIS
# ---------------------------------------------------------------------------
with tab_detect:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Analyze an SMS")
        st.write(
            "Paste a message below. The tuned Naive Bayes model estimates "
            "whether it resembles spam in the training data."
        )
        examples = {
            "Choose an example…": "",
            "Likely spam example": (
                "URGENT! Congratulations, you won a free prize. "
                "Click https://example.com to claim your reward now!"
            ),
            "Likely legitimate example": (
                "Hey, are we still meeting at the library at 4 pm? "
                "I will be there in ten minutes."
            ),
            "Bank-themed example": (
                "Your account needs verification. Click this link "
                "and enter your password immediately."
            ),
        }
        chosen = st.selectbox("Try a sample", list(examples.keys()))
        message = st.text_area(
            "Message text",
            value=examples[chosen],
            height=155,
            placeholder="Paste the SMS message here…",
            max_chars=5000,
        )
        analyze = st.button(
            "Analyze message", type="primary", use_container_width=True
        )

    with right:
        st.subheader("What you'll receive")
        st.markdown("- A prediction and estimated spam probability")
        st.markdown("- Naive Bayes-specific feature contributions")
        st.markdown("- Text-based warning signs to review")
        st.markdown("- A reminder to verify suspicious requests independently")
        st.info(
            "A legitimate-looking message can still be a scam, and a spam-like "
            "message can be legitimate. Check the sender and never share OTPs or passwords."
        )

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
                st.caption(
                    "This is a model score, not a guarantee or a calibrated measure "
                    "of real-world fraud risk."
                )

            st.subheader("Naive Bayes feature contributions")
            spam_terms, ham_terms = get_nb_prediction_terms(model, message, top_n=5)
            explanation_left, explanation_right = st.columns(2)
            with explanation_left:
                st.markdown("**Terms pushing toward spam**")
                if spam_terms:
                    for term, contribution in spam_terms:
                        st.write(f"- `{term}` — contribution: +{contribution:.3f}")
                else:
                    st.write("No present TF-IDF terms push the score toward spam.")
            with explanation_right:
                st.markdown("**Terms pushing toward legitimate**")
                if ham_terms:
                    for term, contribution in ham_terms:
                        st.write(f"- `{term}` — contribution: {contribution:.3f}")
                else:
                    st.write("No present TF-IDF terms push the score toward legitimate.")
            st.caption(
                "Contributions use TF-IDF values multiplied by the difference between "
                "Naive Bayes class-conditional log feature probabilities. They describe "
                "learned associations and are not proof of fraud or safety."
            )

            st.subheader("Potential warning signs")
            if signs:
                for sign in signs:
                    st.markdown(f"- ⚠️ {sign}")
            else:
                st.write(
                    "No warning signs from the app's simple text rules were detected. "
                    "This does not prove the message is safe."
                )

            if prediction == "spam" or signs:
                st.warning(
                    "Safety tip: don't click unexpected links or share passwords, PINs, "
                    "or OTPs. Verify through the organization's official app or website."
                )
            else:
                st.info("Safety tip: check the sender and context before trusting any message.")


# ---------------------------------------------------------------------------
# TAB 2: MODEL EVALUATION, CV TUNING AND COMPARISON
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Evaluation on held-out test data")
    st.write(
        "These metrics are calculated on the 20% test set, which was not used "
        "for hyperparameter tuning."
    )
    nb = metrics["naive_bayes"]
    lr = metrics["logistic"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NB accuracy", f"{nb['accuracy']:.2%}")
    c2.metric("NB spam precision", f"{nb['precision']:.2%}")
    c3.metric("NB spam recall", f"{nb['recall']:.2%}")
    c4.metric("NB spam F1-score", f"{nb['f1']:.2%}")

    st.caption(
        "Spam precision: among messages predicted as spam, how many were spam. "
        "Spam recall: among actual spam messages, how many were detected. "
        "F1 balances spam precision and recall."
    )

    st.subheader("Confusion matrix — Tuned Multinomial Naive Bayes")
    nb_cm_df = pd.DataFrame(
        nb["confusion"],
        index=["Actual legitimate", "Actual spam"],
        columns=["Predicted legitimate", "Predicted spam"],
    )
    st.dataframe(nb_cm_df, width="stretch")
    st.caption("Rows are actual labels; columns are model predictions.")

    st.divider()
    st.subheader("Model comparison")
    st.write(
        "Both models use the same held-out test set and comparable TF-IDF settings. "
        "Naive Bayes alpha was selected using cross-validation on training data only."
    )
    comparison = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Spam precision",
                "Spam recall",
                "Spam F1-score",
                "Training / tuning time (seconds)",
            ],
            "Tuned Multinomial Naive Bayes": [
                nb["accuracy"], nb["precision"], nb["recall"], nb["f1"],
                metrics["tuning_time"],
            ],
            "Logistic Regression": [
                lr["accuracy"], lr["precision"], lr["recall"], lr["f1"],
                metrics["logistic_training_time"],
            ],
        }
    )
    display_comparison = comparison.copy()
    for column in ["Tuned Multinomial Naive Bayes", "Logistic Regression"]:
        display_comparison[column] = display_comparison.apply(
            lambda row: (
                f"{row[column]:.3f} seconds"
                if row["Metric"] == "Training / tuning time (seconds)"
                else f"{row[column]:.2%}"
            ),
            axis=1,
        )
    st.dataframe(display_comparison, width="stretch")

    metric_chart = pd.DataFrame(
        {
            "Tuned Multinomial Naive Bayes": [
                nb["accuracy"], nb["precision"], nb["recall"], nb["f1"]
            ],
            "Logistic Regression": [
                lr["accuracy"], lr["precision"], lr["recall"], lr["f1"]
            ],
        },
        index=["Accuracy", "Spam precision", "Spam recall", "Spam F1-score"],
    )
    st.subheader("Visual comparison")
    st.bar_chart(metric_chart)
    st.caption(
        "Choose based on measured performance and the application's goals. "
        "A higher accuracy alone does not necessarily mean better spam detection."
    )

    st.divider()
    st.subheader("Naive Bayes hyperparameter tuning")
    st.write(f"**Selected alpha:** {metrics['best_alpha']}")
    st.write(f"**Best 5-fold cross-validation spam F1-score:** {metrics['best_cv_f1']:.2%}")
    st.write(f"**Time spent tuning:** {metrics['tuning_time']:.3f} seconds")
    st.write(
        "Alpha controls smoothing in Multinomial Naive Bayes. The selected value "
        "maximizes mean cross-validation spam F1 on the training split."
    )
    cv_display = metrics["cv_results"].copy()
    cv_display["Mean CV spam F1"] = cv_display["Mean CV spam F1"].map(lambda x: f"{x:.2%}")
    cv_display["CV F1 standard deviation"] = cv_display["CV F1 standard deviation"].map(lambda x: f"{x:.3%}")
    st.dataframe(cv_display, width="stretch")

    with st.expander("Tuned Naive Bayes classification report"):
        st.code(nb["report"])
    with st.expander("Logistic Regression confusion matrix and report"):
        lr_cm_df = pd.DataFrame(
            lr["confusion"],
            index=["Actual legitimate", "Actual spam"],
            columns=["Predicted legitimate", "Predicted spam"],
        )
        st.dataframe(lr_cm_df, width="stretch")
        st.code(lr["report"])

    st.divider()
    st.subheader("Dataset overview")
    d1, d2, d3 = st.columns(3)
    d1.metric("Unique messages", f"{metrics['total']:,}")
    d2.metric("Training examples", f"{metrics['train_size']:,}")
    d3.metric("Test examples", f"{metrics['test_size']:,}")
    class_df = pd.DataFrame(
        {
            "Class": ["Legitimate (ham)", "Spam"],
            "Messages": [metrics["ham_count"], metrics["spam_count"]],
        }
    ).set_index("Class")
    st.bar_chart(class_df)


# ---------------------------------------------------------------------------
# TAB 3: METHODOLOGY AND LIMITATIONS
# ---------------------------------------------------------------------------
with tab_method:
    st.subheader("The machine-learning pipeline")
    st.markdown(
        """
        **1. Dataset:** Labelled legitimate and spam SMS messages from the UCI repository.

        **2. Cleaning:** Remove invalid rows and duplicate messages.

        **3. Train-test split:** Use a stratified 80/20 split, keeping the test set
        separate from hyperparameter selection.

        **4. Feature extraction:** TF-IDF converts message text into numerical
        word-unigram and bigram features. The vectorizer is fitted within each
        training fold during cross-validation to prevent feature leakage.

        **5. Hyperparameter tuning:** Test Multinomial Naive Bayes alpha values
        0.1, 0.5, 1.0 and 2.0 using 5-fold StratifiedKFold cross-validation.
        Select the value with the highest mean spam F1-score.

        **6. Comparison:** Compare tuned Multinomial Naive Bayes with Logistic
        Regression on the same held-out test set.

        **7. Explainability:** Estimate Naive Bayes feature contributions using
        TF-IDF feature values and class-conditional log feature probabilities.

        **8. Warning signs:** Show separate rule-based indicators. These are not
        the model's explanation and can produce false alarms.
        """
    )

    st.subheader("Limitations and future scope")
    st.markdown(
        """
        - The dataset is historical and English-language; performance on modern,
          multilingual or region-specific scams may differ.
        - Test-set results do not guarantee performance on unseen real-world messages.
        - Model probabilities may not be calibrated to real-world fraud risk.
        - Naive Bayes feature contributions show learned associations, not causation.
        - Rule-based warning signs are indicators, not proof of fraud.
        - Future work could evaluate more recent datasets, character n-grams,
          calibration, drift monitoring and external URL reputation services.
        """
    )

    st.subheader("Dataset citation")
    st.markdown(
        "Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. "
        "UCI Machine Learning Repository. "
        "[Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). "
        "DOI: 10.24432/C5CC84."
    )

st.divider()
st.markdown(
    '<p class="small-note">ScamSentry AI • Educational prototype • '
    "Always verify suspicious messages independently.</p>",
    unsafe_allow_html=True,
)