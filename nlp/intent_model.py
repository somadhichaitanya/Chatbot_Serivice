import os, joblib
from typing import Tuple

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

class IntentModel:
    def __init__(self):
        self.model_path = os.path.join(MODEL_DIR, 'intent_clf.joblib')
        if os.path.exists(self.model_path):
            self.clf = joblib.load(self.model_path)
        else:
            # Lazy-fit a tiny default model if not trained yet
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC
            from sklearn.pipeline import Pipeline
            X = ['hi', 'hello', 'track order', 'refund status', 'return policy', 'bye']
            y = ['greet', 'greet', 'track_order', 'refund_status', 'return_policy', 'goodbye']
            self.clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())]).fit(X, y)

    def predict(self, text: str) -> Tuple[str, float]:
        label = self.clf.predict([text])[0]
        # Use decision function margin as a pseudo-confidence
        if hasattr(self.clf.named_steps['clf'], 'decision_function'):
            import numpy as np
            margins = self.clf.named_steps['clf'].decision_function(self.clf.named_steps['tfidf'].transform([text]))
            if margins.ndim == 1:
                conf = float(abs(margins[0]))
            else:
                conf = float(abs(margins.max()))
            # Rescale to 0-1
            conf = 1.0 - 1.0 / (1.0 + conf)
        else:
            conf = 0.5
        return label, conf
