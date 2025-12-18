from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.assessment_matrix = None
        self.assessment_index: List[int] = []

    def fit(self, assessment_texts: Iterable[str], index: Iterable[int]):
        self.assessment_index = list(index)
        self.assessment_matrix = self.vectorizer.fit_transform(assessment_texts)

    def compute(self, job_text: str) -> pd.Series:
        if self.assessment_matrix is None:
            raise ValueError("SimilarityEngine not fitted. Call fit first.")
        job_vec = self.vectorizer.transform([job_text])
        scores = cosine_similarity(job_vec, self.assessment_matrix).flatten()
        return pd.Series(scores, index=self.assessment_index)

    def batch_compute(self, job_texts: List[str]) -> np.ndarray:
        if self.assessment_matrix is None:
            raise ValueError("SimilarityEngine not fitted. Call fit first.")
        job_vecs = self.vectorizer.transform(job_texts)
        return cosine_similarity(job_vecs, self.assessment_matrix)
