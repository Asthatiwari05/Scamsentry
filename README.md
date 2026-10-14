# 🚨 ScamSentry AI
### Explainable SMS Spam Detection Using NLP and Machine Learning

ScamSentry AI is a machine learning-based web application that classifies SMS messages as **Ham (Legitimate)** or **Spam** using Natural Language Processing (NLP) and Machine Learning algorithms.

The application uses **Multinomial Naive Bayes** as its primary classification model and compares its performance with **Logistic Regression**. It also provides model evaluation metrics and approximate word-level explanations to help users understand predictions.

---

## 🌐 Live Demo

**Try ScamSentry AI here:**

🔗 [Launch ScamSentry AI](https://scamsentry-kgarbjbwgskkt3jppqxay2.streamlit.app/))

---

## ✨ Features

- 📩 **SMS Spam Detection:** Classifies messages as spam or legitimate.
- 🤖 **Machine Learning:** Uses Multinomial Naive Bayes for classification.
- ⚙️ **Hyperparameter Tuning:** Uses GridSearchCV to select the best smoothing parameter (`alpha`).
- 📊 **Model Comparison:** Compares Multinomial Naive Bayes with Logistic Regression.
- 📈 **Performance Evaluation:** Displays accuracy, precision, recall, F1-score, confusion matrices, and classification reports.
- 🔍 **Explainable Predictions:** Provides approximate word-level contributions to help interpret predictions.
- ⚠️ **Spam Warning Signs:** Highlights potentially suspicious characteristics of a message.
- 🖥️ **Interactive Dashboard:** Provides a user-friendly interface built with Streamlit.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|:---|:---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| pandas | Data handling and preprocessing |
| scikit-learn | Machine learning and model evaluation |
| TF-IDF | Text feature extraction |
| Multinomial Naive Bayes | Primary classification model |
| Logistic Regression | Comparison model |
| GridSearchCV | Hyperparameter tuning |
| UCI SMS Spam Collection | Dataset |

---

## ⚙️ How It Works

The application follows these steps:

1. **Dataset Loading:** Loads the SMS Spam Collection dataset.
2. **Data Preprocessing:** Validates the data, removes exact duplicate messages, and prepares the dataset for training.
3. **Train-Test Split:** Splits the dataset into training and testing sets using stratified sampling.
4. **Feature Extraction:** Converts text messages into numerical representations using TF-IDF.
5. **Model Training:** Trains Multinomial Naive Bayes and Logistic Regression models.
6. **Hyperparameter Tuning:** Uses five-fold stratified cross-validation to select the Naive Bayes `alpha` parameter based on the spam-class F1-score.
7. **Model Evaluation:** Evaluates both models on a held-out test dataset.
8. **Spam Prediction:** Uses the tuned Naive Bayes model to classify user-entered SMS messages.
9. **Explanation:** Displays approximate word-level contributions and potential warning signs.

---

## 📊 Machine Learning Approach

### 1. TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical features that machine learning models can process.

The application uses:

- Unigrams: Individual words.
- Bigrams: Pairs of consecutive words.
- English stop-word filtering.
- Feature limits to manage the vocabulary size.

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is the primary classification algorithm.

It estimates the probability that a message belongs to a class based on the words present in the message.

The model's `alpha` parameter is tuned using GridSearchCV to identify a suitable smoothing value.

### 3. Logistic Regression

Logistic Regression serves as a comparison model. Its performance is evaluated using the same train-test split to make the comparison more consistent.

### 4. Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted spam messages that are actually spam.
- **Recall:** Proportion of actual spam messages correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications for each class.

The application prioritizes the spam-class F1-score during Naive Bayes hyperparameter tuning.

---

## 📂 Dataset

This project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

🔗 **Dataset Link:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains SMS messages labeled as either `ham` or `spam`.

The application downloads the dataset when it first needs to train the models. Initial startup may take longer because of dataset downloading, model training, and hyperparameter tuning.

---

## 🚀 Run Locally

Follow these steps to run ScamSentry AI on your computer.

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or a compatible version
- Git
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/Asthatiwari05/Scamsentry.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Scamsentry
```

### Step 3: Create a Virtual Environment

**Ubuntu/Linux:**

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. It is usually:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
Scamsentry/
│
├── app.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Files excluded from version control
```

---

## ⚠️ Limitations

- ScamSentry AI is an educational prototype and is not a guaranteed spam detector.
- The model may misclassify messages, including new or sophisticated scam messages.
- The dataset primarily contains historical English-language SMS messages, so performance may vary across languages, regions, and newer scam patterns.
- Word-level explanations are approximate model-based indicators, not definitive proof that a message is fraudulent or legitimate.
- The application analyzes only the text entered by the user. It does not access, read, or block SMS messages on a mobile device.

---

## 🔮 Future Enhancements

- Evaluate the model on newer and more diverse SMS datasets.
- Extend support to additional languages.
- Improve prediction explanations and user guidance.
- Add automated tests for preprocessing and prediction.
- Evaluate model performance on unseen and evolving scam patterns.
- Explore additional machine learning models for comparison.

---

## 🔐 Responsible Use

ScamSentry AI should be used as a supporting tool rather than the sole basis for deciding whether a message is safe.

Always verify suspicious messages through official websites or contact channels. Never share passwords, OTPs, banking credentials, or other sensitive information in response to suspicious messages.

---

## 👨‍💻 Author

**Astha Tiwari**

- GitHub: [@Asthatiwari05](https://github.com/Asthatiwari05)

---

## 📚 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the SMS Spam Collection dataset.
- [Streamlit](https://streamlit.io/) for the web application framework.
- [scikit-learn](https://scikit-learn.org/) for machine learning and model evaluation tools.
- [pandas](https://pandas.pydata.org/) for data processing.

---

<p align="center">
  <b>ScamSentry AI — Making SMS Spam Detection More Understandable.</b>
</p>
