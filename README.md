# 🚨 ScamSentry AI
### Spot the scam. Understand the risk.

**An Explainable SMS Safety and Scam Intelligence Platform**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-ScamSentry-brightgreen?style=for-the-badge)](https://scamsentry-kgarbjbwgskkt3jppqxay2.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/AI-Machine%20Learning-orange?style=flat-square)](https://scikit-learn.org/)

ScamSentry AI is an interactive web application that helps users identify potentially suspicious SMS messages, understand warning signs, compare message variants, and explore potential relationships between scam cases.

It combines machine learning, Natural Language Processing (NLP), explainable predictions, semantic similarity, and graph-based intelligence in one platform.

**🌐 Live Application:** https://scamsentry-kgarbjbwgskkt3jppqxay2.streamlit.app/

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📩 Message Analyzer | Classifies SMS messages as Ham or Spam using Multinomial Naive Bayes. |
| 🧠 Explainable Predictions | Displays approximate word-level contributions and suspicious warning signs. |
| 📊 Model Comparison | Compares Multinomial Naive Bayes and Logistic Regression. |
| 🧬 Mutation Intelligence | Uses TF-IDF and semantic similarity to compare message variants. |
| 🕸️ Scam Genome | Visualizes potential relationships between scam cases using shared indicators. |
| 🛡️ Safety Center | Provides scam awareness information and practical safety guidance. |
| 📥 Report Export | Downloads message similarity comparisons as CSV files. |

---

## 🖥️ Platform Overview

### 📩 Message Analyzer
- Classifies messages as **Ham (Legitimate)** or **Spam**.
- Uses TF-IDF for text feature extraction.
- Tunes the Naive Bayes smoothing parameter with GridSearchCV.
- Compares model performance using standard evaluation metrics.
- Provides approximate explanations for model predictions.

### 🧬 Mutation Intelligence
- Compares multiple messages to find potentially similar variants.
- Uses TF-IDF for lexical similarity.
- Uses Sentence Transformers for semantic similarity.
- Ranks message pairs by similarity score.
- Highlights candidates above a configurable threshold.

### 🕸️ Scam Genome
- Builds a graph of potential relationships between scam cases.
- Uses available indicators such as domains, phone numbers, payment identifiers, and text similarity.
- Helps explore possible connections for further investigation.

### 🛡️ Safety Center
- Provides consumer-oriented scam warnings.
- Explains suspicious message patterns.
- Offers practical guidance for safer decisions.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| pandas | Data processing |
| scikit-learn | Machine learning and model evaluation |
| Multinomial Naive Bayes | Primary SMS classification model |
| Logistic Regression | Comparison model |
| TF-IDF | Text feature extraction and lexical similarity |
| Sentence Transformers | Semantic message similarity |
| PyTorch | Sentence-embedding framework |
| NetworkX | Graph construction and analysis |
| Plotly | Data visualization |
| GridSearchCV | Hyperparameter tuning |
| UCI SMS Spam Collection | SMS classification dataset |

---

## ⚙️ How It Works

### SMS Classification Pipeline

1. Load and validate the SMS Spam Collection dataset.
2. Preprocess the messages and remove exact duplicates.
3. Split the data using stratified sampling.
4. Convert text into numerical features using TF-IDF.
5. Train Multinomial Naive Bayes and Logistic Regression.
6. Tune the Naive Bayes `alpha` parameter using GridSearchCV and five-fold stratified cross-validation.
7. Evaluate the models on a held-out test dataset.
8. Classify user-entered messages and provide approximate explanations.

### Message Similarity Pipeline

1. Accept two or more messages.
2. Calculate lexical similarity using TF-IDF.
3. Generate sentence embeddings for semantic comparison.
4. Rank message pairs by similarity.
5. Highlight candidates that meet the selected threshold.

---

## 📊 Model Evaluation

| Metric | Meaning |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted spam messages that are actually spam |
| Recall | Proportion of actual spam messages correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Breakdown of correct and incorrect classifications |

The classification model prioritizes the spam-class F1-score during Naive Bayes hyperparameter tuning.

---

## 📂 Dataset

The Message Analyzer uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

🔗 [Access the UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains SMS messages labeled as `ham` or `spam`.

Initial loading may take longer because the application needs to obtain the dataset and train or tune its models. Mutation Intelligence may also download a pretrained sentence-embedding model on first use.

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10 or a compatible version supported by the dependencies
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Asthatiwari05/Scamsentry.git
