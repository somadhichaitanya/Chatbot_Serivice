
import json, os, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_intents():
    with open(os.path.join(DATA_DIR, 'intents.json'), 'r') as f:
        data = json.load(f)
    texts, labels = [], []
    for intent in data.get('intents', []):
        label = intent.get('name')
        for ex in intent.get('examples', []):
            texts.append(ex)
            labels.append(label)
    return texts, labels

def main():
    X, y = load_intents()
    if len(X) == 0:
        print("No training data found in intents.json — creating a tiny default model.")
        X = ['hi', 'hello', 'track order', 'refund status', 'return policy', 'bye']
        y = ['greet', 'greet', 'track_order', 'refund_status', 'return_policy', 'goodbye']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ('clf', LinearSVC())
    ])

    # Safely split: attempt stratify, fallback without stratify if dataset is small or stratify fails
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    pipeline.fit(X_train, y_train)

    print('Evaluation on holdout:')
    try:
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print("Could not evaluate on holdout:", e)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'intent_clf.joblib'))
    print("Saved intent_clf.joblib")

    # Fit FAQ vectorizer on questions
    faq_path = os.path.join(DATA_DIR, 'faq.csv')
    if os.path.exists(faq_path):
        df = pd.read_csv(faq_path)
        if 'question' in df.columns:
            vect = TfidfVectorizer(ngram_range=(1,2), min_df=1).fit(df['question'].astype(str).tolist())
            joblib.dump(vect, os.path.join(MODEL_DIR, 'faq_vectorizer.joblib'))
            print('Saved FAQ vectorizer.')

if __name__ == '__main__':
    main()
