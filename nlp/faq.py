import os, joblib, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

class FAQRetriever:
    def __init__(self, threshold: float = 0.35):
        self.df = None
        self.vectorizer = None
        faq_path = os.path.join(DATA_DIR, 'faq.csv')
        if os.path.exists(faq_path):
            self.df = pd.read_csv(faq_path)
            vect_path = os.path.join(MODEL_DIR, 'faq_vectorizer.joblib')
            if os.path.exists(vect_path):
                self.vectorizer = joblib.load(vect_path)
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer().fit(self.df['question'].astype(str).tolist())
        self.threshold = threshold

    def search(self, query: str):
        if self.df is None or self.vectorizer is None or len(self.df)==0:
            return None, 0.0
        q_vec = self.vectorizer.transform([query])
        mtx = self.vectorizer.transform(self.df['question'].astype(str).tolist())
        sims = cosine_similarity(q_vec, mtx)[0]
        if sims.max() >= self.threshold:
            idx = sims.argmax()
            return self.df.iloc[idx]['answer'], float(sims[idx])
        return None, float(sims.max()) if len(sims)>0 else 0.0
