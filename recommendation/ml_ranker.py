import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class RankerModel:
    def __init__(self, model: Optional[GradientBoostingClassifier] = None):
        self.model = model or GradientBoostingClassifier()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "RankerModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(model=model)
