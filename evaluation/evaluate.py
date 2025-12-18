"""
Weak-supervision training + scenario evaluation script.
Run: python evaluation/evaluate.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Add parent directory to path so imports work from evaluation/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from text_preprocessing.text_preprocess import add_combined_text_column, normalize_text
from text_preprocessing.feature_engineering import build_feature_matrix
from recommendation.rule_engine import RuleEngine
from recommendation.similarity_engine import SimilarityEngine
from recommendation.ml_ranker import RankerModel

CATALOG_PATH = "data/shl_catalog.json"
RULES_PATH = "config/rules.yaml"
MODEL_PATH = "models/ranking_model.pkl"


SCENARIOS = [
    {"job_role": "Account Manager", "job_level": "Mid-Professional", "skills": ["client communication", "stakeholder management", "project coordination"], "hiring_stage": "Screening"},
    {"job_role": ".NET Developer", "job_level": "Mid-Professional", "skills": [".NET framework", "C#", "MVC"], "hiring_stage": "Screening"},
    {"job_role": "Cashier", "job_level": "Entry-Level", "skills": ["customer service", "transaction handling", "accuracy"], "hiring_stage": "Screening"},
    {"job_role": "Accounts Payable Clerk", "job_level": "Entry-Level", "skills": ["attention to detail", "invoice processing", "reconciliation"], "hiring_stage": "Interview"},
    {"job_role": "Administrative Assistant", "job_level": "Entry to Mid", "skills": ["administration", "organisation", "time management"], "hiring_stage": "Screening"},
    {"job_role": "Agency Manager", "job_level": "Mid-Professional", "skills": ["leadership", "client relationship management", "operations"], "hiring_stage": "Interview"},
    {"job_role": "Branch Manager", "job_level": "Mid-Professional", "skills": ["leadership", "commercial acumen", "people management"], "hiring_stage": "Interview"},
]


def load_catalog() -> pd.DataFrame:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Handle nested structure and extract products array
    if "shl_product_catalog" in raw:
        data = raw["shl_product_catalog"]["products"]
    else:
        data = raw
    df = pd.DataFrame(data)
    # Convert skills string to list
    if "skills" in df.columns:
        df["skills"] = df["skills"].apply(lambda x: [s.strip() for s in x.split(",")] if isinstance(x, str) else x)
    return add_combined_text_column(df)


def generate_weak_labels(
    scenarios: List[Dict],
    df: pd.DataFrame,
    rule_engine: RuleEngine,
    sim_engine: SimilarityEngine,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_rows: List[np.ndarray] = []
    labels: List[int] = []

    for scenario in scenarios:
        job_text = normalize_text(" ".join([scenario.get("job_role", ""), " ".join(scenario.get("skills", []))]))
        ruled_df = rule_engine.apply(scenario, df)
        rule_scores = ruled_df["rule_score"]
        sim_scores = sim_engine.compute(job_text)

        features = build_feature_matrix(scenario, ruled_df, sim_scores, rule_scores)
        feature_rows.extend(features)

        sorted_idx = sim_scores.sort_values(ascending=False).index.tolist()
        cutoff = max(1, int(len(sorted_idx) * 0.2))
        positives = set(sorted_idx[:cutoff])
        for idx in ruled_df.index:
            labels.append(1 if idx in positives else 0)

    return np.array(feature_rows), np.array(labels)


def train_model(X: np.ndarray, y: np.ndarray, model_path: str = MODEL_PATH) -> RankerModel:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RankerModel()
    model.train(X_train, y_train)
    val_pred = model.predict_proba(X_val)
    auc = roc_auc_score(y_val, val_pred) if len(set(y_val)) > 1 else None
    model.save(model_path)
    print(f"Saved model to {model_path}. Validation AUC: {auc}")
    return model


def evaluate_scenarios(model: RankerModel, df: pd.DataFrame, rule_engine: RuleEngine, sim_engine: SimilarityEngine):
    for scenario in SCENARIOS:
        job_text = normalize_text(" ".join([scenario.get("job_role", ""), " ".join(scenario.get("skills", []))]))
        ruled_df = rule_engine.apply(scenario, df)
        rule_scores = ruled_df["rule_score"]
        sim_scores = sim_engine.compute(job_text)
        features = build_feature_matrix(scenario, ruled_df, sim_scores, rule_scores)
        ml_scores = model.predict_proba(features)
        final_scores = 0.6 * ml_scores + 0.4 * sim_scores.values
        top_indices = np.argsort(final_scores)[::-1][:3]
        print("Scenario:", scenario)
        for pos in top_indices:
            row = ruled_df.iloc[pos]
            print(f"  -> {row['assessment_name']} | final={final_scores[pos]:.3f} sim={sim_scores.iloc[pos]:.3f} rule={rule_scores.iloc[pos]:.3f}")
        print()


def main():
    df = load_catalog()
    rule_engine = RuleEngine(RULES_PATH)
    sim_engine = SimilarityEngine()
    sim_engine.fit(df["combined_text"], df.index)

    X, y = generate_weak_labels(SCENARIOS, df, rule_engine, sim_engine)
    print(f"Generated weak labels: {X.shape} features, positives={int(y.sum())}, negatives={len(y) - int(y.sum())}")

    model = train_model(X, y)
    evaluate_scenarios(model, df, rule_engine, sim_engine)


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()
