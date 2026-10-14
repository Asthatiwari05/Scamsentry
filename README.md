ScamSentry AI
Explainable SMS Spam Detection Using NLP and Machine Learning
ScamSentry AI is a web-based application that classifies SMS messages as
ham (legitimate) or spam using Natural Language Processing (NLP)
and machine learning. It also provides model evaluation results and
approximate feature-based explanations to help users understand which
words may have influenced a prediction.
🌐 Live Demo: Add your deployed Streamlit app link
here
Replace the URL above with your actual Streamlit Community Cloud URL.

Features
- SMS classification: Predicts whether a message is ham or spam.
- Naive Bayes model tuning: Uses GridSearchCV to select a
  suitable smoothing parameter (alpha) based on spam-class F1 score.
- Model comparison: Compares Multinomial Naive Bayes with Logistic
  Regression.
- Performance evaluation: Displays accuracy, precision, recall, F1
  score, confusion matrices, and classification reports.
- Prediction explanations: Shows approximate word-level
  contributions for the Naive Bayes prediction, alongside common
  warning signs.
- Interactive dashboard: Built with Streamlit, with sections for
  model evaluation, message prediction, and methodology.
Tech Stack
- Language: Python
- App framework: Streamlit
- Data processing: pandas
- Machine learning: scikit-learn
- Text representation: TF-IDF with unigrams and bigrams
- Dataset: UCI SMS Spam Collection
How It Works
1. Loads and validates the SMS Spam Collection dataset.
2. Removes exact duplicate messages and prepares a stratified
   training/test split.
3. Converts message text into numerical TF-IDF features.
4. Tunes the Multinomial Naive Bayes alpha parameter using five-fold
   stratified cross-validation, selecting the model by spam-class F1
   score.
5. Trains Logistic Regression as a comparison model.
6. Evaluates both models on a held-out test set.
7. Uses the tuned Naive Bayes model to classify text entered by the
   user.
Dataset
The project uses the SMS Spam Collection dataset from the UCI
Machine Learning Repository:
- Dataset page:
  https://archive.ics.uci.edu/dataset/228/sms+spam+collection
The app downloads the dataset when it first needs to train the models.
Initial startup may therefore take longer than later interactions.
Run Locally
1. Clone the repository
git clone https://github.com/Asthatiwari05/Scamsentry.git
cd Scamsentry
2. Create and activate a virtual environment
Ubuntu/Linux:
python3 -m venv .venv
source .venv/bin/activate
Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Launch the app
streamlit run app.py
Open the local URL shown in your terminal, usually
http://localhost:8501.
Project Structure
Scamsentry/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
Limitations
- This is an educational prototype, not a guaranteed spam detector.
- Predictions depend on the dataset and patterns learned during
  training; new or unusual scams may be misclassified.
- The dataset primarily contains historical English-language SMS
  messages, so performance may differ for other languages, regions, or
  modern scam patterns.
- Word-level explanations are approximate model-based indicators, not
  proof that a message is fraudulent or legitimate.
- The app analyzes only the message text entered by the user; it does
  not access, read, or block messages on a user's phone.
Future Improvements
- Evaluate the model on newer and more diverse SMS datasets.
- Add support for additional languages.
- Improve explanation quality and user-facing guidance.
- Add automated tests and monitor performance on new data.
Responsible Use
Do not rely on this tool as the sole basis for deciding whether a
message, link, or sender is safe. Avoid clicking suspicious links or
sharing passwords, OTPs, or financial information. Verify important
messages through the organization's official website or contact channel.
Author
Astha Tiwari
- GitHub: https://github.com/Asthatiwari05
If you use or build on this project, please credit the UCI SMS Spam
Collection dataset and the libraries used.
