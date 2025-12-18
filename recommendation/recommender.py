import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from text_preprocessing.text_preprocess import add_combined_text_column, normalize_text
from text_preprocessing.feature_engineering import build_feature_matrix
from recommendation.rule_engine import RuleEngine
from recommendation.similarity_engine import SimilarityEngine
from recommendation.ml_ranker import RankerModel


class Recommender:
    def __init__(
        self,
        catalog_path: str,
        rules_path: str,
        model_path: str,
        top_n: int = 5,
        ml_weight: float = 0.6,
        sim_weight: float = 0.4,
    ):
        self.catalog_path = catalog_path
        self.rules_path = rules_path
        self.model_path = model_path
        self.top_n = top_n
        self.ml_weight = ml_weight
        self.sim_weight = sim_weight

        self.catalog_df = self._load_catalog()
        self.catalog_df = add_combined_text_column(self.catalog_df)
        self.rule_engine = RuleEngine(rules_path)
        self.sim_engine = SimilarityEngine()
        self.sim_engine.fit(self.catalog_df["combined_text"], self.catalog_df.index)

        self.model = None
        if Path(model_path).exists():
            self.model = RankerModel.load(model_path)

    def _load_catalog(self) -> pd.DataFrame:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
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
        return df

    def recommend(self, job_req: Dict) -> List[Dict]:
        job_text = normalize_text(
            " ".join(
                [
                    job_req.get("job_role", ""),
                    " ".join(job_req.get("skills", []) or []),
                    job_req.get("hiring_stage", ""),
                ]
            )
        )

        # Rules
        ruled_df = self.rule_engine.apply(job_req, self.catalog_df)
        rule_scores = ruled_df["rule_score"]

        # Similarity
        sim_scores = self.sim_engine.compute(job_text)

        # Features
        features = build_feature_matrix(job_req, ruled_df, sim_scores, rule_scores)

        # ML scoring
        if self.model:
            ml_scores = self.model.predict_proba(features)
        else:
            ml_scores = sim_scores.values  # fallback to similarity when model missing

        # Combine scores
        final_scores = self.ml_weight * ml_scores + self.sim_weight * sim_scores.values

        results = []
        for position, (idx, row) in enumerate(ruled_df.iterrows()):
            results.append(
                {
                    "assessment_name": row["assessment_name"],
                    "final_score": float(final_scores[position]),
                    "explanation": {
                        "rule_score": float(rule_scores[idx]),
                        "similarity_score": float(sim_scores[idx]),
                        "ml_score": float(ml_scores[position]),
                        "rule_pass": bool(row["rule_pass"]),
                        **row["rule_explanation"],
                    },
                }
            )

        sorted_results = sorted(results, key=lambda r: r["final_score"], reverse=True)
        return sorted_results[: self.top_n]
