import os, joblib
from typing import Tuple, Dict, Any, List

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Try to import transformers zero-shot classifier
_zero_shot_available = False
try:
    from transformers import pipeline
    _zero_shot_available = True
except Exception:
    pipeline = None

class AdvancedNLP:
    """
    - Uses transformers zero-shot classification if installed (facebook/bart-large-mnli).
    - Falls back to the existing sklearn pipeline (intent_clf.joblib) if available.
    - Falls back to simple keyword heuristics if nothing else available.
    """
    def __init__(self, intents_list: List[str] = None):
        self.intent_list = intents_list or [
            'greet','goodbye','thanks','track_order','cancel_order','refund_status',
            'return_policy','shipping_info','payment_issue','product_info',
            'exchange_request','complaint','escalate'
        ]
        self.zero_shot = None
        if _zero_shot_available:
            try:
                self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            except Exception:
                self.zero_shot = None

        self.fallback = None
        try:
            self.fallback = joblib.load(os.path.join(MODEL_DIR, 'intent_clf.joblib'))
        except Exception:
            self.fallback = None

    def classify_intent(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        text = text.strip()
        # 1) Zero-shot if available
        if self.zero_shot:
            try:
                res = self.zero_shot(text, self.intent_list)
                label = res['labels'][0]
                score = float(res['scores'][0])
                return label, score, {'scores': dict(zip(res['labels'], res['scores']))}
            except Exception:
                pass

        # 2) Fallback to sklearn pipeline
        if self.fallback:
            try:
                label = self.fallback.predict([text])[0]
                conf = 0.6
                try:
                    clf = self.fallback.named_steps.get('clf')
                    vec = self.fallback.named_steps.get('tfidf')
                    if clf is not None and vec is not None and hasattr(clf, 'decision_function'):
                        Xv = vec.transform([text])
                        margins = clf.decision_function(Xv)
                        import numpy as np
                        if hasattr(margins, 'ndim') and margins.ndim == 1:
                            conf = float(abs(margins[0]))
                        else:
                            conf = float(abs(margins.max()))
                        conf = 1.0 - 1.0 / (1.0 + conf)
                except Exception:
                    conf = 0.6
                return label, conf, {}
            except Exception:
                pass

        # 3) Keyword fallback
        low = text.lower()
        keywords = {
            'track_order': ['track','where is my order','order status','where is my package'],
            'refund_status': ['refund','refunded','refund status'],
            'greet': ['hi','hello','hey'],
            'goodbye': ['bye','goodbye'],
            'escalate': ['human','representative','agent','escalate']
        }
        for lab, kws in keywords.items():
            for kw in kws:
                if kw in low:
                    return lab, 0.6, {}
        return 'unknown', 0.0, {}
