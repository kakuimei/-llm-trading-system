from typing import List, Tuple, Optional
import numpy as np

class NewsRetriever:
    """Simple TF-IDF (if available) or lexical fallback retriever over news strings."""
    def __init__(self, use_tfidf: bool = True):
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.corpus: List[str] = []
        self.index_dates: List[str] = []
        try:
            if use_tfidf:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        except Exception:
            self.vectorizer = None

    def fit(self, docs: List[str], dates: Optional[List[str]] = None):
        self.corpus = [d or "" for d in docs]
        self.index_dates = dates or [""] * len(self.corpus)
        if self.vectorizer is not None:
            self.matrix = self.vectorizer.fit_transform(self.corpus)
        else:
            self.matrix = None

    def topk(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        if not self.corpus:
            return []
        if self.vectorizer is not None and self.matrix is not None:
            qv = self.vectorizer.transform([query or ""]).toarray()[0]
            M = self.matrix.toarray()
            # cosine similarity
            denom = (np.linalg.norm(qv) * np.linalg.norm(M, axis=1) + 1e-9)
            sims = (M @ qv) / denom
        else:
            # lexical overlap fallback
            qs = set((query or "").lower().split())
            sims = np.array([len(qs.intersection(set((c or "").lower().split()))) for c in self.corpus], dtype=float)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def get_context(self, query: str, k: int = 3) -> List[str]:
        pairs = self.topk(query, k)
        return [self.corpus[i] for i, _ in pairs]
