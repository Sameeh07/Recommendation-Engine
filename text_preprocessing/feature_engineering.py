from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_skill_overlap(user_skills: List[str], assessment_skills: List[str]) -> Tuple[int, float]:
    user_set = {s.lower() for s in user_skills}
    assess_set = {s.lower() for s in assessment_skills}
    overlap = user_set.intersection(assess_set)
    overlap_count = len(overlap)
    ratio = overlap_count / len(user_set) if user_set else 0.0
    return overlap_count, ratio


def compute_features(
    job_req: Dict,
    assessment_row: pd.Series,
    similarity_score: float,
    rule_score: float,
) -> List[float]:
    user_skills = job_req.get("skills", []) or []
    assessment_skills = assessment_row.get("skills", []) or []
    overlap_count, overlap_ratio = compute_skill_overlap(user_skills, assessment_skills)

    job_level_input = (job_req.get("job_level") or "").lower()
    job_level_assess = (assessment_row.get("job_level") or "").lower()
    job_level_match = 1 if job_level_input and job_level_input in job_level_assess else 0

    category_input = (job_req.get("category") or "").lower()
    category_assess = (assessment_row.get("category") or "").lower()
    category_match = 1 if category_input and category_input == category_assess else 0

    duration_penalty = 0.0
    hiring_stage = (job_req.get("hiring_stage") or "").lower()
    duration = assessment_row.get("duration_minutes") or 0
    if hiring_stage == "screening" and duration and duration > 30:
        duration_penalty = -0.2
    elif hiring_stage == "interview" and duration and duration > 45:
        duration_penalty = -0.1

    return [
        overlap_count,
        overlap_ratio,
        similarity_score,
        job_level_match,
        category_match,
        duration_penalty,
        rule_score,
    ]


def build_feature_matrix(
    job_req: Dict,
    df: pd.DataFrame,
    similarity_scores: pd.Series,
    rule_scores: pd.Series,
) -> np.ndarray:
    feature_rows = []
    for idx, row in df.iterrows():
        sim = float(similarity_scores.get(idx, 0.0))
        rule = float(rule_scores.get(idx, 0.0))
        feature_rows.append(compute_features(job_req, row, sim, rule))
    return np.array(feature_rows)
