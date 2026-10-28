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

st.markdown(

    """

    <style>

    [data-testid="stSidebarNav"] {

        display: none;

    }

    </style>

    """,

    unsafe_allow_html=True,

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
        <p>
            Explainable SMS spam detection powered by
            Natural Language Processing and machine learning.
        </p>
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

# TAB 1: MESSAGE SAFETY — REDESIGNED USER EXPERIENCE

# ---------------------------------------------------------------------------



with tab_detect:

    st.markdown("## Check a suspicious message")

    st.write(

        "Paste an SMS message to see how the classifier interprets it, "

        "which words influence its prediction, and what you should check next."

    )



    examples = {

        "Choose a sample message": "",

        "Prize notification": (

            "URGENT! Congratulations, you won a free prize. "

            "Click https://example.com to claim your reward now!"

        ),

        "Everyday conversation": (

            "Hey, are we still meeting at the library at 4 pm? "

            "I will be there in ten minutes."

        ),

        "Account verification request": (

            "Your account needs verification. Click this link "

            "and enter your password immediately."

        ),

        "Delivery fee request": (

            "Your parcel is delayed. Pay a delivery fee using "

            "this link to receive your package."

        ),

    }



    def load_sample():

        st.session_state["message_input"] = examples[

            st.session_state["sample_message"]

        ]



    if "message_input" not in st.session_state:

        st.session_state["message_input"] = ""



    if "sample_message" not in st.session_state:

        st.session_state["sample_message"] = "Choose a sample message"



    st.markdown("#### Message to review")



    st.selectbox(

        "Try an example",

        options=list(examples.keys()),

        key="sample_message",

        on_change=load_sample,

    )



    with st.container(border=True):

        message = st.text_area(

            "Message text",

            key="message_input",

            height=180,

            max_chars=5000,

            placeholder=(

                "Paste the suspicious SMS here...\n\n"

                "You can remove personal details before checking."

            ),

            label_visibility="visible",

        )



        st.caption(

            f"{len(message)} / 5,000 characters · "

            "Avoid entering passwords, OTPs, account numbers, "

            "or other sensitive personal information."

        )



        analyze = st.button(

            "🔎 Analyze message",

            type="primary",

            use_container_width=True,

        )



        st.caption(
            "Your message is analyzed using the locally running application. "
            "The current analysis does not verify the sender or open links."
        )



    if analyze:

        if not message.strip():

            st.warning("Enter a message or choose a sample first.")

        else:

            with st.spinner("Analyzing message signals..."):

                prediction = model.predict([message])[0]

                probabilities = model.predict_proba([message])[0]

                classes = list(model.classes_)

                spam_probability = float(

                    probabilities[classes.index("spam")]

                )

                signs = warning_signs(message)

                spam_terms, ham_terms = get_nb_prediction_terms(

                    model, message, top_n=5

                )



            st.divider()

            st.markdown("## Analysis results")

            st.caption(

                "Your message has been analyzed. Review the assessment, "

                "understand the signals, and choose your next step."

            )



            result_col, score_col = st.columns([1.25, 1])



            with result_col:

                with st.container(border=True):

                    if prediction == "spam":

                        st.markdown("### ⚠️ Spam-like message")

                        st.write(

                            "The model found that this message resembles "

                            "spam examples in its training data."

                        )

                    else:

                        st.markdown(

                            "### 🟢 Classified as legitimate-like"

                        )

                        st.write(

                            "The model found that this message resembles "

                            "legitimate examples in its training data."

                        )



                    if signs:

                        st.caption(

                            f"{len(signs)} rule-based warning sign(s) detected"

                        )

                    else:

                        st.caption(

                            "No warning signs from the current text rules detected"

                        )



            with score_col:

                with st.container(border=True):

                    st.markdown("### Model score")

                    st.metric(

                        "Estimated spam probability",

                        f"{spam_probability:.1%}",

                    )

                    st.progress(

                        max(0, min(100, round(spam_probability * 100)))

                    )

                    st.caption(

                        "This score is not a calibrated real-world fraud "

                        "probability. It does not prove a message is safe "

                        "or fraudulent."

                    )



            st.markdown("### What should you review?")



            if signs:

                for sign in signs:

                    st.markdown(f"- ⚠️ **{sign}**")

            else:

                st.write(

                    "The current rule-based checks did not detect any "

                    "listed warning signs. Scams can still use other tactics."

                )

            with st.expander("Understand the model's word-level signals"):

                explain_col1, explain_col2 = st.columns(2)



                with explain_col1:

                    st.markdown("**Terms associated with spam**")

                    if spam_terms:

                        for term, contribution in spam_terms:

                            st.write(f"`{term}` · +{contribution:.3f}")

                    else:

                        st.write("No present terms pushed toward spam.")



                with explain_col2:

                    st.markdown("**Terms associated with legitimate messages**")

                    if ham_terms:

                        for term, contribution in ham_terms:

                            st.write(f"`{term}` · {contribution:.3f}")

                    else:

                        st.write(

                            "No present terms pushed toward legitimate."

                        )



                st.caption(

                    "These values describe learned word associations. "

                    "They do not establish the sender's intent."

                )



            st.markdown("### What should you do next?")



            if signs or prediction == "spam":

                st.warning(

                    "**Pause before acting.** Don't click unexpected links, "

                    "send money, or share verification codes. Verify the request "

                    "through the organization's official app, website, or "

                    "published contact details."

                )

            else:

                st.info(

                    "**Stay cautious.** No listed warning signs were detected, "

                    "but this does not guarantee safety. Independently verify "

                    "unexpected requests before acting."

                )



            with st.expander("Technical details and model signals"):

                st.write(f"Model prediction: `{prediction}`")

                st.write(f"Rule-based warning signs detected: {len(signs)}")

                st.write(

                    "Model: tuned Multinomial Naive Bayes with TF-IDF features."

                )



            st.caption(

                "ScamSentry provides screening assistance, not a definitive "

                    "fraud verdict. Similarity scores and model predictions "

                    "cannot independently establish fraud."

                )



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
